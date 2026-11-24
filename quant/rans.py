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
