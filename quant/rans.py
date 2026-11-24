"""Static, row-class-context rANS for the int8 ECQ codes (encoder side; decoder/decode.py holds the matching decoder).

rANS: Duda 2013 (arXiv:1311.2540), byte-oriented form as in F. Giesen's rans_byte; here 32-bit states, 16-bit
renormalisation words and PB = 12-bit probabilities. Each code tensor (rows already in the row-key sorted order the
artifact stores) is cut into L lanes of `seg` consecutive codes; all lanes are coded in lock step, so both encoder and
decoder are plain vectorised integer numpy (no compiler, deterministic). The context of a code is the class of its
row = the rank of the row's 1-byte spread key among the tensor's distinct keys (the decoder reads the same keys), so
each class has its own frequency table: rows of similar spread share a distribution (the static analogue of zstd's
per-block Huffman tables, but at fractional-bit precision).

Blob layout (little-endian): int32 L, seg, K; K x (int16 smin, int16 nsym, uint16 freq[nsym]); uint32 state[L];
uint16 words[...]: grouped by step (decode order), within a step in lane order, so the decoder reads them with
one running pointer (no per-lane lengths: 4 bytes of overhead per lane, ~0.25% at seg 4096).
"""

import numpy as np

PB = 12
M = 1 << PB
LOW = 1 << 16
SEG = 4096


def row_classes(key_sorted: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense class id per stored row (rows are in key-sorted order) and the number of classes."""
    uniq, inv = np.unique(key_sorted, return_inverse=True)
    return inv.astype(np.int32), len(uniq)


def normalise(cnt: np.ndarray) -> np.ndarray:
    """counts [256] -> frequencies summing to M, every used symbol >= 1."""
    tot = int(cnt.sum())
    f = np.floor(cnt.astype(np.float64) * M / tot).astype(np.int64)
    f[(cnt > 0) & (f == 0)] = 1
    diff = M - int(f.sum())
    while diff != 0:
        if diff > 0:
            f[int(np.argmax(f))] += diff
            diff = 0
        else:
            # take from the symbols with the largest frequencies first (smallest relative cost)
            order = np.argsort(-f, kind="stable")
            for s in order:
                if diff == 0:
                    break
                take = min(-diff, int(f[s]) - 1)
                if take > 0:
                    f[s] -= take
                    diff += take
    return f


def encode(codes_sorted: np.ndarray, key_sorted: np.ndarray) -> bytes:
    """codes_sorted int8 [R, C] in stored row order, key_sorted uint8 [R] (the stored row keys, sorted) -> blob."""
    R, C = codes_sorted.shape
    n = R * C
    sym = (codes_sorted.astype(np.int16) + 128).reshape(-1)
    rcls, K = row_classes(key_sorted)
    cnt = np.bincount(np.repeat(rcls, C) * 256 + sym, minlength=K * 256).reshape(K, 256)
    F = np.zeros((K, 256), dtype=np.int64)
    head = []
    for k in range(K):
        F[k] = normalise(cnt[k])
        used = np.nonzero(F[k])[0]
        smin, smax = int(used[0]), int(used[-1])
        head.append(np.array([smin, smax - smin + 1], dtype=np.int16).tobytes() + F[k, smin : smax + 1].astype(np.uint16).tobytes())
    CUM = np.zeros((K, 256), dtype=np.int64)
    CUM[:, 1:] = np.cumsum(F, axis=1)[:, :-1]
    seg = min(SEG, n)
    L = -(-n // seg)
    lanes = np.arange(L, dtype=np.int64) * seg
    x = np.full(L, LOW, dtype=np.uint64)
    words = np.zeros((seg, L), dtype=np.uint16)
    emitted = np.zeros((seg, L), dtype=bool)
    for i in range(seg - 1, -1, -1):
        idx = lanes + i
        valid = idx < n
        idc = np.minimum(idx, n - 1)
        s = sym[idc]
        k = rcls[idc // C]
        f = F[k, s].astype(np.uint64)
        c = CUM[k, s].astype(np.uint64)
        emit = valid & (x >= (f << np.uint64(20)))  # x_max = ((LOW >> PB) << 16) * f
        words[i] = (x & np.uint64(0xFFFF)).astype(np.uint16)
        emitted[i] = emit
        x = np.where(emit, x >> np.uint64(16), x)
        x = np.where(valid, ((x // f) << np.uint64(PB)) + (x % f) + c, x)
    stream = words[emitted]  # row-major over (step, lane): step ascending, lanes in order = the decoder's read order
    return (np.array([L, seg, K], dtype=np.int32).tobytes() + b"".join(head) + x.astype(np.uint32).tobytes()
            + stream.astype(np.uint16).tobytes())


def estimate_bytes(codes_sorted: np.ndarray, key_sorted: np.ndarray) -> int:
    """Ideal static-model size (bits from the normalised tables) + table and lane overhead, without coding."""
    R, C = codes_sorted.shape
    n = R * C
    sym = (codes_sorted.astype(np.int16) + 128).reshape(-1)
    rcls, K = row_classes(key_sorted)
    cnt = np.bincount(np.repeat(rcls, C) * 256 + sym, minlength=K * 256).reshape(K, 256)
    bits = 0.0
    tables = 0
    for k in range(K):
        if cnt[k].sum() == 0:
            continue
        f = normalise(cnt[k])
        used = cnt[k] > 0
        bits += float((cnt[k][used] * (PB - np.log2(f[used]))).sum())
        nz = np.nonzero(f)[0]
        tables += 4 + 2 * (int(nz[-1]) - int(nz[0]) + 1)
    L = -(-n // min(SEG, n))
    return int(bits / 8) + tables + 4 * L + 12


# ------------------------------------------------------------------ 4-state dependent quantisation (quant/dq.py)
# Codes n of a trellis tensor are coded as their quantisation index k given the trellis state (Q0: n = 2k; Q1:
# n = 2k - sgn k; next state = DQ_NEXT[state][k & 1]). Columns in trellis order (perm), rows in natural order,
# flattened row-major; the state restarts at 0 every CH codes (CH divides C: whole row chunks), and a lane holds m
# whole chunks (m = max(1, SEG // CH)), so the state is known wherever a lane starts and all lanes run in lock step.
# Context = (row class, state) with NS = 4 tables per class, or (row class, quantiser) with NS = 2 (flag bit 2) when
# that is smaller; row class = rank of the row's 1-byte spread key (stored in natural row order).
# Blob layout (little-endian): int32 DQ_MAGIC, R, C, K, flags (bit 0: perm present, bit 1: perm is uint32, bit 2:
# NS = 2, bit 3: order key instead of perm), m, tlen, CH; perm [C] (uint16 or uint32) if present, or the order key as
# (C + 1) // 2 bytes of 4-bit nibbles (low nibble first; perm = stable argsort of the key); tables: one zstd frame of
# tlen bytes holding K * NS x (int16 smin, int16 nsym, uint16 freq[nsym]; nsym 0 = unused); uint32 state[L];
# uint16 words[...] grouped by step (decode order), lanes in order.
DQ_MAGIC = 0x34515144  # b"DQQ4"
DQ_NEXT = np.array([[0, 2], [2, 0], [1, 3], [3, 1]], dtype=np.int64)


def dq_symbols(codes_t: np.ndarray, chunk: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """codes in trellis order int [R, C] -> (state [R, C], symbol k + 128 [R, C]); the state restarts every `chunk`
    columns (0 = once per row). ValueError if the codes are not a trellis path."""
    n = codes_t.astype(np.int64)
    R, C = n.shape
    st = np.zeros(R, dtype=np.int64)
    S = np.empty((R, C), dtype=np.int64)
    sym = np.empty((R, C), dtype=np.int64)
    for j in range(C):
        if chunk and j % chunk == 0:
            st[:] = 0
        q = st >> 1
        nj = n[:, j]
        odd = (nj & 1) == 1
        if not np.all(np.where(q == 0, ~odd, odd | (nj == 0))):
            raise ValueError("codes are not a 4-state trellis path")
        k = (nj + q * np.sign(nj)) >> 1  # exact: nj + q sgn nj is even
        S[:, j] = st
        sym[:, j] = k + 128
        st = DQ_NEXT[st, k & 1]
    return S, sym


def _dq_prep(codes: np.ndarray, perm: np.ndarray | None, key: np.ndarray, chunk: int):
    ct = codes if perm is None else codes[:, perm]
    S, sym = dq_symbols(ct, chunk)
    _, cls = np.unique(key, return_inverse=True)
    cls = cls.astype(np.int64).reshape(-1)
    return S, sym, cls, int(cls.max()) + 1


def _dq_counts(S, sym, cls, K, ns):
    st = S if ns == 4 else (S >> 1)
    ctx = cls[:, None] * ns + st
    cnt = np.bincount((ctx * 256 + sym).reshape(-1), minlength=K * ns * 256).reshape(K * ns, 256)
    return ctx, cnt


def _dq_tables(cnt: np.ndarray) -> tuple[np.ndarray, bytes]:
    F = np.zeros_like(cnt, dtype=np.int64)
    head = []
    for c in range(cnt.shape[0]):
        if cnt[c].sum() == 0:
            head.append(np.array([0, 0], dtype=np.int16).tobytes())
            continue
        F[c] = normalise(cnt[c])
        used = np.nonzero(F[c])[0]
        smin, smax = int(used[0]), int(used[-1])
        head.append(np.array([smin, smax - smin + 1], dtype=np.int16).tobytes() + F[c, smin : smax + 1].astype(np.uint16).tobytes())
    import zstandard

    return F, zstandard.ZstdCompressor(level=19).compress(b"".join(head))


def _dq_ideal(cnt: np.ndarray) -> float:
    bits = 0.0
    for c in range(cnt.shape[0]):
        if cnt[c].sum():
            f = normalise(cnt[c])
            used = cnt[c] > 0
            bits += float((cnt[c][used] * (PB - np.log2(f[used]))).sum())
    return bits


def _dq_chunk(C: int, chunk: int) -> int:
    ch = C if not chunk else chunk
    if ch <= 0 or C % ch:
        raise ValueError("trellis chunk must divide the column count")
    return ch


def encode_dq(codes: np.ndarray, perm: np.ndarray | None, key: np.ndarray, ns: int | None = None, okey: np.ndarray | None = None,
              chunk: int = 0) -> bytes:
    """codes int8 [R, C] in natural order, perm [C] (trellis column order) or None, key uint8 [R] natural row order.
    okey: uint8 [C] bucket key < 16 with perm = stable argsort(okey) (stored as 4-bit nibbles instead of perm).
    chunk: trellis restart period (0 = per row). ns: 4 (per state) or 2 (per quantiser) tables per row class;
    None = the smaller ideal size."""
    R, C = codes.shape
    ch = _dq_chunk(C, chunk)
    if okey is not None:
        okey = np.asarray(okey, dtype=np.uint8)
        if okey.max(initial=0) > 15:
            raise ValueError("order key must be < 16")
        p2 = np.argsort(okey, kind="stable")
        if perm is not None and not np.array_equal(p2, np.asarray(perm)):
            raise ValueError("perm is not the stable argsort of the order key")
        perm = p2
    S, sym, cls, K = _dq_prep(codes, perm, key, ch)
    if ns is None:
        c4 = _dq_counts(S, sym, cls, K, 4)
        c2 = _dq_counts(S, sym, cls, K, 2)
        e4 = _dq_ideal(c4[1]) / 8 + len(_dq_tables(c4[1])[1])
        e2 = _dq_ideal(c2[1]) / 8 + len(_dq_tables(c2[1])[1])
        ns, (ctx, cnt) = (4, c4) if e4 <= e2 else (2, c2)
    else:
        ctx, cnt = _dq_counts(S, sym, cls, K, ns)
    F, tab = _dq_tables(cnt)
    CUM = np.zeros_like(F)
    CUM[:, 1:] = np.cumsum(F, axis=1)[:, :-1]
    m = max(1, SEG // ch)
    seg = m * ch
    n = R * C
    L = -(-n // seg)
    fsym = sym.reshape(-1)
    fctx = ctx.reshape(-1)
    lanes = np.arange(L, dtype=np.int64) * seg
    x = np.full(L, LOW, dtype=np.uint64)
    words = np.zeros((seg, L), dtype=np.uint16)
    emitted = np.zeros((seg, L), dtype=bool)
    for i in range(seg - 1, -1, -1):
        idx = lanes + i
        valid = idx < n
        idc = np.minimum(idx, n - 1)
        f = F[fctx[idc], fsym[idc]].astype(np.uint64)
        c = CUM[fctx[idc], fsym[idc]].astype(np.uint64)
        emit = valid & (x >= (f << np.uint64(20)))
        words[i] = (x & np.uint64(0xFFFF)).astype(np.uint16)
        emitted[i] = emit
        x = np.where(emit, x >> np.uint64(16), x)
        x = np.where(valid, ((x // f) << np.uint64(PB)) + (x % f) + c, x)
    flags = 4 if ns == 2 else 0
    pblob = b""
    if okey is not None:
        flags |= 8
        k2 = np.zeros(2 * ((C + 1) // 2), dtype=np.uint8)
        k2[:C] = okey
        pblob = (k2[0::2] | (k2[1::2] << 4)).astype(np.uint8).tobytes()
    elif perm is not None:
        wide = C > 65535
        flags |= 1 | (2 if wide else 0)
        pblob = np.asarray(perm, dtype=np.uint32 if wide else np.uint16).tobytes()
    return (np.array([DQ_MAGIC, R, C, K, flags, m, len(tab), ch], dtype=np.int32).tobytes() + pblob + tab
            + x.astype(np.uint32).tobytes() + words[emitted].astype(np.uint16).tobytes())


def estimate_dq_bytes(codes: np.ndarray, perm: np.ndarray | None, key: np.ndarray, ns: int = 4, okey: bool = False, chunk: int = 0) -> int:
    """Ideal size of encode_dq (static tables, compressed table bytes) without coding."""
    R, C = codes.shape
    ch = _dq_chunk(C, chunk)
    S, sym, cls, K = _dq_prep(codes, perm, key, ch)
    _, cnt = _dq_counts(S, sym, cls, K, ns)
    seg = max(1, SEG // ch) * ch
    perm_bytes = 0 if perm is None else ((C + 1) // 2 if okey else C * (4 if C > 65535 else 2))
    return int(_dq_ideal(cnt) / 8) + len(_dq_tables(cnt)[1]) + 4 * (-(-(R * C) // seg)) + 32 + perm_bytes
