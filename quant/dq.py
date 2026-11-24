"""4-state dependent quantisation (trellis-coded quantisation, TCQ) on the ECQ grid, inside the GPTQ block loop.

Two scalar quantisers on one step D per output row (Schwarz, Nguyen, Marpe, Wiegand, "Hybrid video coding with
trellis-coded quantization", DCC 2019; the dependent quantisation of H.266/VVC):
  Q0 reconstructs 2k D           (even multiples of D),
  Q1 reconstructs (2k - sgn k) D  (zero and odd multiples of D).
A 4-state machine picks the quantiser of the next weight: states 0, 1 use Q0, states 2, 3 use Q1, and
next = NEXT[state][k & 1] with NEXT = [[0, 2], [2, 0], [1, 3], [3, 1]]. Each quantiser alone has spacing 2D, but the
encoder chooses among paths through the union (all multiples of D) with a Viterbi search, which packs the
reconstruction points like a lattice (Marcellin & Fischer 1990: TCQ gains ~1 dB of the 1.53 dB space-filling loss of
scalar quantisation). The decoder only needs the parity of the decoded index to follow the state.

What is stored: the int8 code n = rec / D (n = 2k in Q0, n = 2k - sgn k in Q1), so dequantisation is unchanged
(w = bf16(n * step[row])) and any decoder that reads n decodes these weights. The gain only shows with a coder that
knows the state: quant/rans.py encode_dq codes k with one table per (row class, state), see decoder/decode.py.

The search runs on the exact GPTQ objective (Frantar et al. 2023, arXiv:2210.17323): with U the upper Cholesky factor
of H^-1, (w - w_hat) H (w - w_hat)^T = sum_i ((w~_i - w_hat_i) / U_ii)^2, where w~_i is weight i after the error
feedback of the columns before it. Each of the 4 survivor paths of a row keeps its own feedback-updated copy of the
current column block, so the path metric is that sum exactly (plus lambda x bits of the coded index); at the end of
every block the best survivor is kept, its codes are replayed through the ordinary GPTQ block update, and the next
block starts from that survivor's state. Rows are independent, so everything is batched over rows (GPU or CPU).
"""

import logging
import math

import torch

from quant.ecq import UResult


log = logging.getLogger("quant.dq")
NEXT = [[0, 2], [2, 0], [1, 3], [3, 1]]
# predecessors of next state t: (state, parity of k) pairs with NEXT[state][parity] == t
PRED_S = [[0, 1], [2, 3], [0, 1], [2, 3]]
PRED_P = [[0, 1], [0, 1], [1, 0], [1, 0]]
NMAX = (126, 127)  # largest |n| of Q0 (even) and Q1 (odd): codes stay int8
KOFF = 64  # rate tables are indexed by k + KOFF, |k| <= 64
NK = 2 * KOFF + 1


def k_of(n: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Quantisation index of reconstruction n (integer-valued tensor) in quantiser q (0/1, broadcastable)."""
    return torch.div(n + q * torch.sign(n), 2, rounding_mode="floor")


class _Tables:
    def __init__(self, bits: torch.Tensor | None, cls: torch.Tensor | None, lam: float):
        # bits [K, 4, NK] (-log2 p), cls [R] -> per-row [R, 4, NK]
        self.lam = lam
        self.rows = None if (bits is None or lam <= 0) else (lam * bits[cls]).float()

    def rate(self, rows: slice | None, k: torch.Tensor) -> torch.Tensor | float:
        """k [R, 4, C] integer -> lam * bits [R, 4, C] (state = dim 1)."""
        if self.rows is None:
            return 0.0
        t = self.rows if rows is None else self.rows[rows]
        idx = (k.long() + KOFF).clamp(0, NK - 1)
        return torch.gather(t, 2, idx)


class _Trellis:
    """Static-shape state of the trellis search for blocks of B columns over R rows, so the per-column step can be
    replayed from a CUDA graph (one launch per column instead of ~40 small kernels; Python dispatch dominates the
    search otherwise). The same step runs eagerly on the CPU or when graphs are off (QUANT_ECQ_DQ_GRAPH=0)."""

    def __init__(self, R: int, B: int, device, feedback: bool, rated: bool):
        f32 = torch.float32
        self.R, self.B, self.feedback, self.rated, self.device = R, B, feedback, rated, device
        self.W = torch.zeros((R, 4, B), device=device, dtype=f32)  # per survivor: block weights after its feedback
        self.S = torch.ones((R, B), device=device, dtype=f32)  # grid (step x column factor)
        self.h = torch.zeros(B, device=device, dtype=f32)  # column weights 1/U_jj^2 (normalised)
        self.U = torch.zeros((B, B), device=device, dtype=f32)  # strictly upper part of the block's U
        self.Ud = torch.ones(B, device=device, dtype=f32)  # diagonal of the block's U
        self.cost = torch.zeros((R, 4), device=device, dtype=f32)
        self.rate = torch.zeros((R, 4, NK), device=device, dtype=f32) if rated else None
        self.j = torch.zeros(1, dtype=torch.long, device=device)
        self.back_s = torch.zeros((B, R, 4), dtype=torch.uint8, device=device)
        self.back_n = torch.zeros((B, R, 4), dtype=torch.int16, device=device)
        self.off = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=f32).view(1, 4)
        self.offk = self.off.unsqueeze(2)
        nm = torch.tensor([NMAX[0], NMAX[0], NMAX[1], NMAX[1]], device=device, dtype=f32).view(1, 4, 1).expand(1, 4, 3)
        self.nmax, self.nmin = nm.contiguous(), (-nm).contiguous()
        pred_sp = torch.tensor([[a * 2 + b for a, b in zip(PRED_S[t], PRED_P[t])] for t in range(4)], device=device)
        self.pred_flat = pred_sp.view(-1)
        self.pred_exp = pred_sp.view(1, 4, 2).expand(R, 4, 2).contiguous()
        self.zeros = torch.zeros((R, 4), device=device, dtype=f32)
        self.inf = torch.full((R, 4, 3), float("inf"), device=device, dtype=f32)
        self.graph = None

    def step(self) -> None:
        R, B, j = self.R, self.B, self.j
        sj = self.S.index_select(1, j)  # [R, 1]
        x = self.W.index_select(2, j).view(R, 4) / sj  # [R, 4] in steps
        lo = torch.floor((x - self.off) * 0.5) * 2.0 + self.off  # largest level <= x of the state's quantiser (exact integer)
        n = torch.stack([lo, lo + 2.0, self.zeros], dim=2)
        n = torch.minimum(torch.maximum(n, self.nmin), self.nmax)
        k = torch.div(n + self.offk * torch.sign(n), 2, rounding_mode="floor")
        odd = torch.remainder(k, 2.0) > 0.5
        d = x.unsqueeze(2) - n
        tot = torch.addcmul(self.cost.unsqueeze(2), d * self.h.index_select(0, j), d)
        if self.rated:
            tot = tot + torch.gather(self.rate, 2, (k.long() + KOFF).clamp(0, NK - 1))
        tv = torch.stack([torch.where(odd, self.inf, tot), torch.where(odd, tot, self.inf)], dim=2)  # [R, 4 s, 2 p, 3]
        bv, bi = tv.min(dim=3)
        cost, o = bv.view(R, 8).index_select(1, self.pred_flat).view(R, 4, 2).min(dim=2)
        sp = self.pred_exp.gather(2, o.unsqueeze(2)).view(R, 4)  # s * 2 + p of the kept predecessor of each state
        s_star = torch.div(sp, 2, rounding_mode="floor")
        n_star = n.view(R, 12).gather(1, s_star * 3 + bi.view(R, 8).gather(1, sp))
        self.cost.copy_(cost)
        self.back_s.index_copy_(0, j, s_star.to(torch.uint8).unsqueeze(0))
        self.back_n.index_copy_(0, j, n_star.to(torch.int16).unsqueeze(0))
        if self.feedback:
            g = self.W.gather(1, s_star.unsqueeze(2).expand(R, 4, B))
            err = (x.gather(1, s_star) - n_star) * (sj / self.Ud.index_select(0, j))  # (w~ - n s) / U_jj
            torch.addcmul(g, err.unsqueeze(2), self.U.index_select(0, j).view(1, 1, B), value=-1.0, out=self.W)
        self.j.add_(1)

    def _capture(self) -> None:
        st = torch.cuda.Stream(device=self.device)
        st.wait_stream(torch.cuda.current_stream(self.device))
        saved = [t.clone() for t in (self.W, self.cost, self.j)]
        with torch.cuda.stream(st):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream(self.device).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.step()
        for t, v in zip((self.W, self.cost, self.j), saved):
            t.copy_(v)
        self.graph = g

    def run(self, Wb, grid, hcol, Ub, start, rate_rows, use_graph: bool):
        """-> (n [R, B] float, end states [R], state in which each code was coded [R, B] uint8) for one block."""
        R, B = self.R, self.B
        if self.feedback:
            self.W.copy_(Wb.unsqueeze(1).expand(R, 4, B))
            self.U.copy_(torch.triu(Ub, diagonal=1))
            self.Ud.copy_(torch.diagonal(Ub))
        else:
            self.W.copy_(Wb.unsqueeze(1).expand(R, 4, B))
        self.S.copy_(grid if grid.dim() == 2 else grid.unsqueeze(1).expand(R, B))
        self.h.copy_(hcol)
        if self.rated:
            self.rate.copy_(rate_rows)
        self.cost.fill_(float("inf"))
        self.cost.scatter_(1, start.view(-1, 1), 0.0)
        self.j.zero_()
        if use_graph and self.graph is None:
            self._capture()
        for _ in range(B):
            if use_graph:
                self.graph.replay()
            else:
                self.step()
        end = self.cost.argmin(dim=1)
        t = end.clone()
        rows = torch.arange(R, device=self.device)
        out = torch.empty((R, B), device=self.device)
        sts = torch.empty((R, B), device=self.device, dtype=torch.uint8)
        bn, bs = self.back_n.long(), self.back_s.long()
        for jj in range(B - 1, -1, -1):
            out[:, jj] = bn[jj, rows, t].float()
            t = bs[jj, rows, t]
            sts[:, jj] = t.to(torch.uint8)
        if not torch.equal(t, start):
            raise RuntimeError("trellis backtrack did not reach the start state")
        return out, end, sts


_CACHE: dict = {}
_GRAPH_STATE = {"ok": None}  # None = not yet checked on this process, True / False after the first self-check


def _graphs_wanted(device) -> bool:
    import os

    return torch.device(device).type == "cuda" and os.environ.get("QUANT_ECQ_DQ_GRAPH", "1") == "1" and _GRAPH_STATE["ok"] is not False


@torch.no_grad()
def _viterbi_block(Wb, s_row, hcol, Ub, start, tabs: "_Tables", feedback: bool):
    """One block of columns. Wb [R, B] (current, feedback-updated weights), s_row [R] or [R, B] grid, hcol [B] column
    weights 1/U_ii^2 (normalised), Ub [B, B] upper Cholesky block (feedback only), start [R] states.
    Returns n [R, B] (float, integer valued), end states [R] and the state of every code [R, B] (uint8).

    Per column and survivor state s the candidates are the two nearest levels of the state's quantiser and zero;
    for each (s, parity of k) the cheapest candidate goes on, and each next state t takes the cheaper of its two
    predecessors (s, p) with NEXT[s][p] == t. With feedback each survivor carries its own copy of the block's columns
    (gathered on every step, so a merged path keeps its residual)."""
    R, B = Wb.shape
    dev = Wb.device
    rated = tabs.rows is not None
    key = (R, B, str(dev), feedback, rated)
    tr = _CACHE.get(key)
    if tr is None:
        if len(_CACHE) > 16:
            _CACHE.clear()
        tr = _CACHE[key] = _Trellis(R, B, dev, feedback, rated)
    Ub_ = Ub if Ub is not None else torch.zeros((B, B), device=dev)
    graph = _graphs_wanted(dev)
    if graph and _GRAPH_STATE["ok"] is None:
        # first use on this process: the graph replay must give the eager result (else graphs stay off)
        try:
            n_e, e_e, _ = tr.run(Wb, s_row, hcol, Ub_, start, tabs.rows, use_graph=False)
            n_g, e_g, _ = tr.run(Wb, s_row, hcol, Ub_, start, tabs.rows, use_graph=True)
            _GRAPH_STATE["ok"] = bool(torch.equal(n_e, n_g) and torch.equal(e_e, e_g))
        except Exception as exc:  # noqa: BLE001
            log.warning("trellis CUDA graph failed (%s): eager steps", exc)
            _GRAPH_STATE["ok"] = False
        log.info("trellis CUDA graph self-check: %s", "same codes as eager" if _GRAPH_STATE["ok"] else "OFF")
        if not _GRAPH_STATE["ok"]:
            _CACHE.clear()
            tr = _CACHE[key] = _Trellis(R, B, dev, feedback, rated)
        graph = bool(_GRAPH_STATE["ok"])
    if not graph:
        return tr.run(Wb, s_row, hcol, Ub_, start, tabs.rows, use_graph=False)
    try:
        return tr.run(Wb, s_row, hcol, Ub_, start, tabs.rows, use_graph=True)
    except Exception as exc:  # noqa: BLE001 - a capture for a new shape failed: eager from here on
        log.warning("trellis CUDA graph failed for shape %s (%s): eager steps", key, exc)
        _GRAPH_STATE["ok"] = False
        _CACHE.clear()
        tr = _CACHE[key] = _Trellis(R, B, dev, feedback, rated)
        return tr.run(Wb, s_row, hcol, Ub_, start, tabs.rows, use_graph=False)


def _hcol(Ud: torch.Tensor) -> torch.Tensor:
    h = 1.0 / Ud.pow(2)
    return h / h.mean()


def order_key(hdiag: torch.Tensor, buckets: int = 16) -> torch.Tensor:
    """Bucketed act order: bucket b = rank of the column's Hessian diagonal (descending) * buckets // in. The trellis
    column order is the stable sort of b (columns of one bucket stay in natural order), so the decoder rebuilds it
    from log2(buckets) bits per column instead of a full permutation (GPTQ act order, Frantar et al. 2023)."""
    n = hdiag.numel()
    rank = torch.empty(n, dtype=torch.long, device=hdiag.device)
    rank[torch.argsort(hdiag, descending=True, stable=True)] = torch.arange(n, device=hdiag.device)
    return (rank * buckets // max(1, n)).to(torch.uint8)


@torch.no_grad()
def _splits(rows: int, split: list[int] | None):
    if not split:
        return [(0, rows)]
    out, r0 = [], 0
    for n in split:
        out.append((r0, r0 + n))
        r0 += n
    if r0 != rows:
        raise ValueError("split does not cover the rows")
    return out


@torch.no_grad()
def est_bytes(codes_t: torch.Tensor, states: torch.Tensor, okey: bool, perm: bool, chunk: int) -> int:
    """Estimated size of rans.encode_dq for one tensor from its trellis-order codes and states (on their device):
    empirical entropy of k given (row class, state) + tables (~half their raw size after zstd) + lane states + order."""
    R, C = codes_t.shape
    if R * C == 0:
        return 0
    n = codes_t.long()
    s = states.long()
    k = torch.div(n + (s >> 1) * torch.sign(n), 2, rounding_mode="floor")
    spread = n.abs().float().mean(1)
    key = torch.clamp(torch.round(16.0 * torch.log2(spread + 1e-3)) + 128, 0, 255)
    _, cls = torch.unique(key, return_inverse=True)
    nctx = (int(cls.max()) + 1) * 4
    cnt = torch.bincount(((cls.view(-1, 1) * 4 + s) * NK + (k + KOFF).clamp(0, NK - 1)).flatten(), minlength=nctx * NK).double().view(nctx, NK)
    tot = cnt.sum(1, keepdim=True)
    nz = cnt > 0
    bits = float((cnt[nz] * torch.log2((tot.expand_as(cnt))[nz] / cnt[nz])).sum())
    used = nz.any(1)
    first = torch.where(nz, torch.arange(NK, device=cnt.device), NK).min(1).values
    last = torch.where(nz, torch.arange(NK, device=cnt.device), -1).max(1).values
    table = float(((last - first + 1).clamp(min=0) * 2 + 4)[used].sum()) * 0.5
    ch = chunk if chunk else C
    lanes = -(-(R * C) // (max(1, 4096 // ch) * ch))
    order = ((C + 1) // 2 if okey else 2 * C) if perm else 0
    return int(bits / 8 + table + 4 * lanes + 32 + order)


def chunk_for(cols: int, block: int = 128, longest: int = 1024) -> int:
    """Trellis restart period (columns in trellis order): the largest multiple of the GPTQ block <= longest that
    divides cols, else cols (one chain per row). Every chunk starts in state 0, so the decoder can run many short
    lanes in lock step (decode time) at the cost of one forced state per chunk."""
    for ch in range(longest - longest % block, block - 1, -block):
        if ch > 0 and cols % ch == 0:
            return ch
    return cols


@torch.no_grad()
def gptq_dq(weight: torch.Tensor, H: torch.Tensor, step: torch.Tensor, damp: float = 0.01, blocksize: int = 128, act_order: bool | str = True,
            col: torch.Tensor | None = None, lam: float = 0.0, bits: torch.Tensor | None = None, cls: torch.Tensor | None = None,
            chunk: int | None = None, split: list[int] | None = None) -> UResult:
    """GPTQ with a 4-state trellis per row onto step[row] * col[column] * n (n as in the module docstring).
    weight [out, in], H [in, in] fp32 on the compute device. lam > 0 with bits [K, 4, NK] and cls [out] adds
    lam * bits(k | row class, state) to the path metric (units: squared error in steps, columns weighted by 1/U_ii^2
    normalised to mean 1). act_order: True = exact act order (full permutation stored), "b16" = bucketed act order
    (16 buckets, 4 bits per column stored), False = natural order. Returns UResult with dq=True, the column order of
    the trellis (perm, None = natural) and, for bucketed order, its key (okey). chunk: trellis restart period
    (a multiple of blocksize dividing in, or in; default chunk_for(in, blocksize)). split: row counts of stacked
    tensors; UResult.est is then a list of estimated coded sizes per tensor (else an int)."""
    dev = H.device
    W = weight.to(device=dev, dtype=torch.float32).clone()
    out_f, in_f = W.shape
    s_row = step.to(device=dev, dtype=torch.float32)
    c_col = None if col is None else col.to(device=dev, dtype=torch.float32)
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm, okey = None, None
    if act_order == "b16":
        okey = order_key(torch.diag(H), 16)
        perm = torch.argsort(okey, stable=True)
    elif act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
    if perm is not None:
        W = W[:, perm]
        H = H[perm][:, perm]
        if c_col is not None:
            c_col = c_col[perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(in_f, device=dev)
    try:
        L = torch.linalg.cholesky(H)
    except Exception:  # noqa: BLE001
        H += 0.1 * torch.mean(torch.diag(H)) * torch.eye(in_f, device=dev)
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    hcol = _hcol(torch.diag(Hinv))
    tabs = _Tables(None if bits is None else bits.to(dev), None if cls is None else cls.to(dev), lam)
    codes = torch.zeros(out_f, in_f, dtype=torch.int8, device=dev)
    chunk = chunk_for(in_f, blocksize) if chunk is None else chunk
    if chunk != in_f and (chunk % blocksize or in_f % chunk):
        raise ValueError("chunk must be a multiple of the block size dividing the column count")
    state = torch.zeros(out_f, dtype=torch.long, device=dev)
    states = torch.empty((out_f, in_f), dtype=torch.uint8, device=dev)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        if i1 % chunk == 0:
            state = torch.zeros(out_f, dtype=torch.long, device=dev)
        W1 = W[:, i1:i2].clone()
        Hinv1 = Hinv[i1:i2, i1:i2]
        grid = s_row if c_col is None else s_row.unsqueeze(1) * c_col[i1:i2].unsqueeze(0)  # same fp32 product as UResult.grid()
        nb, state, sts = _viterbi_block(W1, grid, hcol[i1:i2], Hinv1, state, tabs, feedback=True)
        codes[:, i1:i2] = nb.to(torch.int8)
        states[:, i1:i2] = sts
        # replay the chosen codes through the ordinary GPTQ block update (exact feedback of the kept path)
        Err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            g = grid if grid.dim() == 1 else grid[:, i]
            err = (W1[:, i] - nb[:, i] * g) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    est = [est_bytes(codes[r0:r1], states[r0:r1], okey is not None, perm is not None, chunk) for r0, r1 in _splits(out_f, split)]
    if perm is not None:
        unperm = torch.empty_like(codes)
        unperm[:, perm] = codes
        codes = unperm
    return UResult(codes.cpu(), step.float().cpu(), None if col is None else col.float().cpu(), dq=True,
                   perm=None if perm is None else perm.cpu().to(torch.int64), okey=None if okey is None else okey.cpu(), chunk=chunk,
                   est=est if split else est[0])


@torch.no_grad()
def rtn_dq(weight: torch.Tensor, step: torch.Tensor, device="cuda", lam: float = 0.0, bits=None, cls=None, blocksize: int = 128,
           row_chunk: int = 1 << 15, chunk: int | None = None) -> UResult:
    """Trellis search without error feedback (each weight's own squared error, natural column order)."""
    W = weight.to(device=device, dtype=torch.float32)
    out_f, in_f = W.shape
    chunk = chunk_for(in_f, blocksize) if chunk is None else chunk
    s_row = step.to(device=device, dtype=torch.float32)
    codes = torch.zeros(out_f, in_f, dtype=torch.int8)
    hcol = torch.ones(in_f, device=device)
    for r0 in range(0, out_f, row_chunk):
        r1 = min(out_f, r0 + row_chunk)
        tabs = _Tables(None if bits is None else bits.to(device), None if cls is None else cls[r0:r1].to(device), lam)
        state = torch.zeros(r1 - r0, dtype=torch.long, device=device)
        for i1 in range(0, in_f, blocksize):
            i2 = min(i1 + blocksize, in_f)
            if i1 % chunk == 0:
                state = torch.zeros(r1 - r0, dtype=torch.long, device=device)
            nb, state, _ = _viterbi_block(W[r0:r1, i1:i2], s_row[r0:r1], hcol[i1:i2], None, state, tabs, feedback=False)
            codes[r0:r1, i1:i2] = nb.to(torch.int8).cpu()
    return UResult(codes, step.float().cpu(), dq=True, perm=None, chunk=chunk)


def states_and_k(codes_t: torch.Tensor, chunk: int | None = None):
    """codes_t int8/any int [R, C] in trellis order -> (states [R, C] the state each code was coded in, k [R, C]);
    the state restarts at 0 every `chunk` columns (None = once per row). Raises ValueError if a code is not in its
    state's quantiser (then the tensor is not a trellis path)."""
    n = codes_t.long()
    R, C = n.shape
    nxt = torch.tensor(NEXT, dtype=torch.long)
    s = torch.zeros(R, dtype=torch.long)
    S = torch.empty((R, C), dtype=torch.long)
    K = torch.empty((R, C), dtype=torch.long)
    for j in range(C):
        if chunk and j % chunk == 0:
            s = torch.zeros(R, dtype=torch.long)
        q = (s >= 2).long()
        nj = n[:, j]
        odd = torch.remainder(nj, 2) == 1
        ok = torch.where(q == 0, ~odd, odd | (nj == 0))
        if not bool(ok.all()):
            raise ValueError("codes are not a 4-state trellis path")
        k = k_of(nj, q)
        S[:, j] = s
        K[:, j] = k
        s = nxt[s, torch.remainder(k, 2)]
    return S, K


def row_class(codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 1-byte spread key of ffcodec (per row) and the dense class id (rank among the distinct keys)."""
    spread = codes.float().abs().mean(1)
    key = torch.clamp(torch.round(16.0 * torch.log2(spread + 1e-3)) + 128, 0, 255).to(torch.uint8)
    _, cls = torch.unique(key, return_inverse=True)
    return key, cls


def rate_tables(codes: torch.Tensor, perm: torch.Tensor | None, smooth: float = 0.5, chunk: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (bits [K, 4, NK] = -log2 p(k | class, state) with add-smooth counts, cls [R]) from a trellis code tensor."""
    ct = codes if perm is None else codes[:, perm]
    S, K = states_and_k(ct, chunk)
    _, cls = row_class(codes)
    nc = int(cls.max()) + 1 if cls.numel() else 1
    idx = (cls.view(-1, 1) * 4 + S) * NK + (K + KOFF).clamp(0, NK - 1)
    cnt = torch.bincount(idx.flatten(), minlength=nc * 4 * NK).float().view(nc, 4, NK) + smooth
    bits = -torch.log2(cnt / cnt.sum(dim=2, keepdim=True))
    return bits, cls


def rate_model(weight: torch.Tensor, step: torch.Tensor, device, cols: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Rate tables for the trellis search: (row class, state) statistics of the trellis RTN codes on the same grid,
    from the first `cols` columns only (None = all; the statistics need far fewer columns than the search)."""
    w = weight if cols is None else weight[:, :cols]
    u = rtn_dq(w, step, device=device)
    return rate_tables(u.codes, None, chunk=u.chunk)


def ideal_bits(codes: torch.Tensor, perm: torch.Tensor | None, per_state: bool = True, chunk: int | None = None) -> float:
    """Empirical conditional entropy (bits) of the tensor's indices given (row class, state) or (row class, quantiser)."""
    ct = codes if perm is None else codes[:, perm]
    S, K = states_and_k(ct, chunk)
    _, cls = row_class(codes)
    ctx = S if per_state else (S >= 2).long()
    nctx = 4 if per_state else 2
    nc = int(cls.max()) + 1
    idx = (cls.view(-1, 1) * nctx + ctx) * NK + (K + KOFF)
    cnt = torch.bincount(idx.flatten(), minlength=nc * nctx * NK).double().view(-1, NK)
    tot = cnt.sum(1, keepdim=True)
    p = cnt / tot.clamp(min=1)
    return float(-(cnt[cnt > 0] * torch.log2(p[cnt > 0])).sum())
