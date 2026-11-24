"""Entropy-coded uniform quantisation (ECQ) for the freeform track.

Every quantised 2-D weight is w = bf16(code * step[row]) with integer codes in [-127, 127] (int8) and one fp32 step per
output row; there is no zero point and no group-wise range. With an entropy coder behind it, a uniform quantiser with
a fixed step is the rate-distortion-efficient scalar choice at moderate rates (Gish & Pierce 1968: within 0.25 bit of
the bound at high rate, independent of the weight distribution), while min/max group ranges spend bits on outliers and
on per-group scales. Rows whose weights are small relative to the step simply produce low-entropy codes.

Bits are spread over tensors / rows by loss sensitivity: with uniform noise of variance step^2/12 on a weight with
(empirical) Fisher f, the loss grows by ~f * step^2 / 24, and the rate falls by log2(step) per weight. Minimising rate
at a fixed total loss increase gives step ~ 1 / sqrt(f) (the same Lagrangian argument as HAWQ-V2, Dong et al. 2020,
arXiv:1911.03852, and the rate allocation in Gersho & Gray, ch. 8). One global constant c scales all steps; it is set
by a secant search on c until the calibration ratio r = loss / loss_bf16 lands just under the knee.

Codes are found by GPTQ (Frantar et al. 2023, arXiv:2210.17323) with the uniform grid in place of the group grid.
"""

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


log = logging.getLogger("quant.ecq")
QMAX = 127
IGNORE_INDEX = -100


@dataclass
class UResult:
    codes: torch.Tensor  # int8 [out, in]
    step: torch.Tensor  # fp32 [out]
    col: torch.Tensor | None = None  # fp32 [in] or None: grid of weight (i, j) is step[i] * col[j]
    dq: bool = False  # codes are a 4-state trellis path (quant/dq.py); dequantisation is the same
    perm: torch.Tensor | None = None  # dq only: column order of the trellis (int64 [in]); None = natural order
    okey: torch.Tensor | None = None  # dq only: uint8 [in] bucket key with perm = stable argsort(okey), or None
    chunk: int = 0  # dq only: trellis restart period in columns (trellis order); 0 = once per row
    est: int | list | None = None  # dq only: estimated size of the state-aware coding (quant/dq.py est_bytes)

    def grid(self) -> torch.Tensor:
        s = self.step.float().unsqueeze(1).to(self.codes.device)
        return s if self.col is None else s * self.col.float().unsqueeze(0).to(self.codes.device)

    def dequant(self, group: int = 0, dtype=torch.bfloat16) -> torch.Tensor:  # group unused (QResult-compatible)
        return (self.codes.float() * self.grid()).to(dtype)


@torch.no_grad()
def rtn_u(weight: torch.Tensor, step: torch.Tensor, device="cuda") -> UResult:
    w = weight.to(device=device, dtype=torch.float32)
    s = step.to(device=device, dtype=torch.float32).unsqueeze(1)
    q = torch.clamp(torch.round(w / s), -QMAX, QMAX).to(torch.int8)
    return UResult(q.cpu(), step.float().cpu())


@torch.no_grad()
def column_factors(H: torch.Tensor, alpha: float, clamp: float) -> torch.Tensor:
    """col[j] ~ (H_jj / geomean H)^(-alpha/2), clamped to [1/clamp, clamp], geometric mean ~1 (fp32 [in], CPU)."""
    d = torch.diag(H).float().clamp(min=1e-12)
    ld = d.log()
    ld = ld - ld.median()
    return torch.exp(-0.5 * alpha * ld).clamp(1.0 / clamp, clamp).cpu()


def floor_step(weight: torch.Tensor, step: torch.Tensor, col: torch.Tensor | None, max_code: float) -> torch.Tensor:
    """step[i] >= max_j |w_ij| / col[j] / max_code: codes stay inside int8 with headroom for GPTQ's updates."""
    w = weight.float().abs()
    if col is not None:
        w = w / col.to(w.device).unsqueeze(0)
    return torch.maximum(step.float().to(w.device), w.amax(1) / max_code).cpu()


def rate_table(weight: torch.Tensor, step: torch.Tensor, device) -> torch.Tensor:
    """bits[code + QMAX] = -log2 p(code), p from this tensor's RTN codes on the same grid (add-0.5 smoothing)."""
    q = torch.clamp(torch.round(weight.to(device=device, dtype=torch.float32) / step.to(device).float().unsqueeze(1)), -QMAX, QMAX)
    hist = torch.bincount((q + QMAX).long().flatten(), minlength=2 * QMAX + 1).float() + 0.5
    return -torch.log2(hist / hist.sum())


def gptq_u(weight: torch.Tensor, H: torch.Tensor, step: torch.Tensor, damp: float = 0.01, blocksize: int = 128, act_order: bool = True,
           col: torch.Tensor | None = None, rdo: float = 0.0) -> UResult:
    """rdo > 0: rate-distortion-optimised rounding. Each weight takes floor or ceil of x = w / grid by the smaller
    (x - q)^2 + rdo * bits(q), bits from the tensor's code histogram (RDOQ as in H.264/HEVC encoders; for uniform
    quantisation at high rate dD/dR = -2 ln2 D, so rdo ~ 2 ln2 / 12 = 0.116 balances them)."""
    """GPTQ onto the grid step[row] * col[column] * {-127..127}. weight [out, in], H [in, in] fp32 on GPU."""
    W = weight.to(device=H.device, dtype=torch.float32).clone()
    out_f, in_f = W.shape
    s_row = step.to(device=H.device, dtype=torch.float32)
    bits = rate_table(W, s_row, H.device) if rdo > 0 else None
    c_col = None if col is None else col.to(device=H.device, dtype=torch.float32)
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = None
    if act_order:  # columns by decreasing Hessian diagonal (free here: no groups tie columns to scales)
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        if c_col is not None:
            c_col = c_col[perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(in_f, device=H.device)
    try:
        L = torch.linalg.cholesky(H)
    except Exception:  # noqa: BLE001
        H += 0.1 * torch.mean(torch.diag(H)) * torch.eye(in_f, device=H.device)
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    codes = torch.zeros(out_f, in_f, dtype=torch.int8, device=H.device)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            s = s_row if c_col is None else s_row * c_col[i1 + i]  # same fp32 product as UResult.grid()
            if bits is None:
                q = torch.clamp(torch.round(w / s), -QMAX, QMAX)
            else:
                x = w / s
                lo = torch.clamp(torch.floor(x), -QMAX, QMAX)
                hi = torch.clamp(lo + 1, -QMAX, QMAX)
                c_lo = (x - lo).pow(2) + rdo * bits[(lo + QMAX).long()]
                c_hi = (x - hi).pow(2) + rdo * bits[(hi + QMAX).long()]
                q = torch.where(c_hi < c_lo, hi, lo)
            codes[:, i1 + i] = q.to(torch.int8)
            err = (w - q * s) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    if perm is not None:
        unperm = torch.empty_like(codes)
        unperm[:, perm] = codes
        codes = unperm
    return UResult(codes.cpu(), step.float().cpu(), None if col is None else col.float().cpu())


def fisher_rows(model, rows, names: list[str], n_rows: int, max_len: int) -> dict[str, torch.Tensor]:
    """Empirical Fisher per output row: mean over rows of sum_j (dL/dW_ij)^2, L = the row's mean supervised CE.

    Returns {name: fp32 [out] on CPU}, the per-weight mean (divided by in_features)."""
    params = dict(model.named_parameters())
    targets = {n: params[n] for n in names if n in params}
    flags = {n: p.requires_grad for n, p in params.items()}
    for p in params.values():
        p.requires_grad_(False)
    for p in targets.values():
        p.requires_grad_(True)
    device = next(model.parameters()).device
    acc = {n: torch.zeros(p.shape[0], device=device, dtype=torch.float32) for n, p in targets.items()}
    used = 0
    try:
        for row in rows:
            if used >= n_rows:
                break
            ids = torch.tensor(row.input_ids[:max_len], device=device).view(1, -1)
            labels = torch.tensor(row.labels[:max_len], device=device)
            tg = labels[1:]
            mask = tg != IGNORE_INDEX
            if not mask.any():
                continue
            hs = model.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state[0, :-1][mask]
            logits = model.get_output_embeddings()(hs).float()
            loss = F.cross_entropy(logits, tg[mask])
            loss.backward()
            for n, p in targets.items():
                if p.grad is not None:
                    acc[n] += p.grad.float().pow(2).sum(1)
                    p.grad = None
            used += 1
    finally:
        for n, p in params.items():
            p.grad = None
            p.requires_grad_(flags[n])
    out = {n: (a / max(1, used) / targets[n].shape[1]).cpu() for n, a in acc.items()}
    log.info("Fisher from %d rows on %d tensors", used, len(out))
    return out


class Allocation:
    """step[name] = c * base[name]; base from Fisher (step ~ 1/sqrt(f)) or from the weights' RMS (uniform bits)."""

    def __init__(self, shapes: dict[str, list[int]], weights_rms: dict[str, torch.Tensor], fisher: dict[str, torch.Tensor] | None,
                 row_level: set[str], row_clamp: float | dict = 4.0):
        self.base: dict[str, torch.Tensor] = {}
        self.numel = {n: math.prod(shapes[n]) for n in weights_rms}
        if fisher is None:
            for n, rms in weights_rms.items():
                self.base[n] = torch.full((shapes[n][0],), float(rms.pow(2).mean().sqrt()) + 1e-12)
            self.kind = "rms"
            return
        self.kind = "fisher"
        tensor_f = {n: float(f.mean()) for n, f in fisher.items()}
        positive = sorted(v for v in tensor_f.values() if v > 0)
        floor = positive[len(positive) // 2] * 1e-4 if positive else 1.0
        for n in weights_rms:
            f = fisher.get(n)
            ft = max(tensor_f.get(n, 0.0), floor)
            b = torch.full((shapes[n][0],), 1.0 / math.sqrt(ft))
            if f is not None and n in row_level:
                rc = row_clamp.get(n, 4.0) if isinstance(row_clamp, dict) else row_clamp
                rel = (ft / f.clamp(min=ft * 1e-6)).sqrt().clamp(1.0 / rc, rc)
                b = b * rel
            self.base[n] = b

    def set_floor(self, absmax: dict[str, torch.Tensor], max_code: float = 100.0) -> None:
        """step >= |w|max(row) / max_code: never clip (GPTQ updates can push a weight past the row's max; headroom)."""
        self.floor = {n: (a.float() / max_code).clamp(min=1e-12) for n, a in absmax.items()}

    def steps(self, c: float) -> dict[str, torch.Tensor]:
        floor = getattr(self, "floor", {})
        return {n: torch.maximum(b * c, floor[n]).float() if n in floor else (b * c).float() for n, b in self.base.items()}

    def c_for_loss(self, delta_loss: float) -> float:
        """Noise-model guess: delta_loss = sum_l n_l f_l (c b_l)^2 / 24 with f_l b_l^2 = 1 -> c = sqrt(24 dL / N)."""
        if self.kind != "fisher":
            return float("nan")
        n = sum(self.numel.values())
        return math.sqrt(24.0 * max(delta_loss, 1e-6) / n)

    def c_for_ratio(self, step_over_rms: float, rms: dict[str, torch.Tensor]) -> float:
        """c such that the numel-weighted geometric mean of step / rms equals step_over_rms."""
        tot = sum(self.numel.values())
        lg = sum(self.numel[n] * math.log(float(self.base[n].mean()) / (float(rms[n].pow(2).mean().sqrt()) + 1e-12)) for n in self.base)
        return step_over_rms / math.exp(lg / tot)


class FakeQuantLinearU(torch.nn.Module):
    """Block-AP on the ECQ grid: latent weights with straight-through rounding onto step[row] * {-127..127}.
    Steps stay fixed unless learn_step (fixed steps keep the rate set by the step search)."""

    def __init__(self, linear, step: torch.Tensor, learn_step: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(linear.weight.detach().float().clone())
        st = step.to(linear.weight.device).float().clone()
        self.step = torch.nn.Parameter(st) if learn_step else None
        self.register_buffer("step_fixed", st)
        self.bias = linear.bias

    def _s(self):
        return (self.step if self.step is not None else self.step_fixed).clamp(min=1e-12).unsqueeze(1)

    def forward(self, x):
        s = self._s()
        z = self.weight / s
        q = torch.clamp(z + (torch.round(z) - z).detach(), -QMAX, QMAX)
        return F.linear(x, (q * s).to(x.dtype), self.bias)

    def qparams(self) -> list:
        return [self.step] if self.step is not None else []

    @torch.no_grad()
    def result(self) -> UResult:
        s = self._s()
        q = torch.clamp(torch.round(self.weight / s), -QMAX, QMAX).to(torch.int8)
        return UResult(q.cpu(), s.squeeze(1).float().cpu())
