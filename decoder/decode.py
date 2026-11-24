"""Freeform decoder: python decode.py /artifact /output

Artifact (written by quant/ffcodec.py): ff-g<N>.json (highest N wins) + payload files it names. Every blob is one
zstd frame. Tensor encodings:
  raw: little-endian bf16 bytes of the tensor.
  q:   asymmetric group-wise integer quantisation along the last dim. codes uint8 [out, in] (one code per byte; zstd's
       entropy coder does the bit packing), scale bf16 [out, in/group], zero uint8 [out, in/group];
       w = bf16((code - zero) * scale), computed in float32 (bit-identical to the trainer's dequant).
  u:   uniform (entropy-coded) quantisation per output row. codes int8 [out, in], step float32 [out];
       w = bf16(code * step[row]), computed in float32. Optional rowkey uint8 [out]: the codes blob holds the rows in
       np.argsort(rowkey, kind="stable") order. Optional col float32 [in]: w = bf16(code * (step[row] * col[column])).
  codes_rans (instead of codes): the same int8 codes coded with static row-class-context rANS (see rans_decode).
  codes_dq (instead of codes): codes of a 4-state dependent quantisation (trellis) path, natural row order, coded with
       rANS given (row class, trellis state) (see dq_decode); rowkey then only gives the row classes.
Output: top-level *.safetensors with exactly the tensors listed, all bf16. Deterministic: no randomness, no threads
racing on shared output (the rANS stage runs in a process pool over tensors; results are consumed in manifest order).
"""

import json
import os
import re
import sys

import numpy as np
import torch
import zstandard
from safetensors.torch import save_file


SHARD_BYTES = 2 * 1024**3


def find_meta(artifact: str) -> str:
    best = None
    for name in os.listdir(artifact):
        m = re.fullmatch(r"ff-g(\d+)\.json", name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), name)
    if best is None:
        raise SystemExit("no ff-g*.json in artifact")
    return os.path.join(artifact, best[1])


class Blobs:
    def __init__(self, artifact: str, files: list[str]):
        self.paths = [os.path.join(artifact, f) for f in files]
        self.dctx = zstandard.ZstdDecompressor()

    def raw(self, ref) -> bytes:
        fi, off, length = ref
        with open(self.paths[fi], "rb") as f:
            f.seek(off)
            data = f.read(length)
        if len(data) != length:
            raise SystemExit(f"short read in {self.paths[fi]}")
        return data

    def get(self, ref, nbytes: int) -> bytearray:
        fi, off, length = ref
        with open(self.paths[fi], "rb") as f:
            f.seek(off)
            data = f.read(length)
        if len(data) != length:
            raise SystemExit(f"short read in {self.paths[fi]}")
        out = bytearray(self.dctx.decompress(data, max_output_size=nbytes))
        if len(out) != nbytes:
            raise SystemExit(f"blob decoded to {len(out)} bytes, expected {nbytes}")
        return out


def bf16_from(buf: bytearray, shape) -> torch.Tensor:
    return torch.frombuffer(buf, dtype=torch.bfloat16).reshape(list(shape))


def dequant(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group: int) -> torch.Tensor:
    """codes uint8 [out, in], scale bf16 [out, G], zero uint8 [out, G] -> bf16 [out, in]."""
    out_f, in_f = codes.shape
    result = torch.empty((out_f, in_f), dtype=torch.bfloat16)
    step = max(1, (1 << 24) // in_f)
    for r in range(0, out_f, step):
        c = codes[r : r + step].float().view(-1, in_f // group, group)
        z = zero[r : r + step].float().unsqueeze(-1)
        s = scale[r : r + step].float().unsqueeze(-1)
        result[r : r + step] = ((c - z) * s).view(-1, in_f).to(torch.bfloat16)
    return result


def dequant_u(codes: torch.Tensor, step: torch.Tensor, col: torch.Tensor | None = None) -> torch.Tensor:
    """codes int8 [out, in], step float32 [out], col float32 [in] or None -> bf16 [out, in];
    w = bf16(code * (step[i] * col[j])) in float32 (the product order the trainer uses)."""
    out_f, in_f = codes.shape
    result = torch.empty((out_f, in_f), dtype=torch.bfloat16)
    rows = max(1, (1 << 24) // in_f)
    for r in range(0, out_f, rows):
        g = step[r : r + rows].unsqueeze(1)
        if col is not None:
            g = g * col.unsqueeze(0)
        result[r : r + rows] = (codes[r : r + rows].float() * g).to(torch.bfloat16)
    return result


def rans_decode(blob: bytes, rows: int, cols: int, key_sorted: np.ndarray) -> np.ndarray:
    """Inverse of quant/rans.py encode: static rANS (Duda 2013), 32-bit states, 16-bit words, 12-bit probabilities,
    L lanes of seg consecutive codes decoded in lock step; context = class (rank among distinct keys) of the code's row;
    renormalisation words are read with one running pointer (grouped by step, lanes in order).
    Integer numpy only (deterministic). Returns int8 [rows, cols] in stored (key-sorted) row order."""
    pb, low = 12, 1 << 16
    lanes, seg, ncls = (int(v) for v in np.frombuffer(blob, dtype=np.int32, count=3))
    pos = 12
    freq = np.zeros((ncls, 256), dtype=np.int64)
    for k in range(ncls):
        smin, nsym = (int(v) for v in np.frombuffer(blob, dtype=np.int16, count=2, offset=pos))
        pos += 4
        freq[k, smin : smin + nsym] = np.frombuffer(blob, dtype=np.uint16, count=nsym, offset=pos)
        pos += 2 * nsym
    cum = np.zeros((ncls, 257), dtype=np.int64)
    cum[:, 1:] = np.cumsum(freq, axis=1)
    if not np.all(cum[:, 256] == 1 << pb):
        raise SystemExit("rANS table does not sum to 4096")
    slot_sym = np.zeros((ncls, 1 << pb), dtype=np.int64)
    for k in range(ncls):
        slot_sym[k] = np.repeat(np.arange(256), freq[k])
    x = np.frombuffer(blob, dtype=np.uint32, count=lanes, offset=pos).astype(np.uint64)
    pos += 4 * lanes
    words = np.frombuffer(blob, dtype=np.uint16, offset=pos).astype(np.uint64)
    ptr = 0  # words are grouped by step, lanes in order within a step
    n = rows * cols
    _, rcls = np.unique(key_sorted, return_inverse=True)
    rcls = rcls.astype(np.int64)
    if int(rcls.max(initial=0)) + 1 != ncls:
        raise SystemExit("rANS class count mismatch")
    out = np.empty(lanes * seg, dtype=np.int64)
    base = np.arange(lanes, dtype=np.int64) * seg
    mask_slot = np.uint64((1 << pb) - 1)
    for i in range(seg):
        idx = base + i
        valid = idx < n
        k = rcls[np.minimum(idx, n - 1) // cols]
        slot = (x & mask_slot).astype(np.int64)
        s = slot_sym[k, slot]
        out[idx] = s
        f = freq[k, s].astype(np.uint64)
        c = cum[k, s].astype(np.uint64)
        x = np.where(valid, f * (x >> np.uint64(pb)) + slot.astype(np.uint64) - c, x)
        need = np.nonzero(valid & (x < np.uint64(low)))[0]
        if len(need):
            x[need] = (x[need] << np.uint64(16)) | words[ptr : ptr + len(need)]
            ptr += len(need)
    if ptr != len(words):
        raise SystemExit("rANS stream not fully consumed")
    return (out[:n] - 128).astype(np.int8).reshape(rows, cols)


DQ_MAGIC = 0x34515144
DQ_NEXT = np.array([[0, 2], [2, 0], [1, 3], [3, 1]], dtype=np.int64)


def dq_decode(blob: bytes, rows: int, cols: int, key: np.ndarray) -> np.ndarray:
    """Inverse of quant/rans.py encode_dq: 4-state dependent quantisation (VVC-style trellis-coded quantisation;
    Schwarz et al., DCC 2019) coded with static rANS (Duda 2013). Codes in trellis column order (perm), rows in
    natural order, flattened row-major; the state restarts at 0 every CH codes, and each of the L lanes holds m whole
    CH-chunks; all lanes are decoded in lock step. States 0, 1 use quantiser Q0 (n = 2k), states 2, 3 use Q1
    (n = 2k - sgn k); next state = DQ_NEXT[state][k & 1]. Context = (row class, state), or (row class, state >> 1)
    when flag bit 2 is set. Per (context, slot) the decoded symbol's frequency, slot offset, output code and next
    state are tabulated once, so a step is a few table lookups. Integer numpy only (deterministic).
    key: uint8 [rows] row keys in natural order. Returns int8 codes n [rows, cols] in natural order."""
    pb, low = 12, 1 << 16
    magic, R, C, ncls, flags, m, tlen, ch = (int(v) for v in np.frombuffer(blob, dtype=np.int32, count=8))
    if magic != DQ_MAGIC or R != rows or C != cols or m < 1 or ch < 1 or C % ch:
        raise SystemExit("dq blob header mismatch")
    pos = 32
    perm = None
    if flags & 8:  # bucket order key, 4-bit nibbles: trellis column order = stable argsort of the key
        nb = (C + 1) // 2
        b = np.frombuffer(blob, dtype=np.uint8, count=nb, offset=pos)
        okey = np.empty(2 * nb, dtype=np.uint8)
        okey[0::2], okey[1::2] = b & 15, b >> 4
        perm = np.argsort(okey[:C], kind="stable").astype(np.int64)
        pos += nb
    elif flags & 1:
        pdt = np.uint32 if flags & 2 else np.uint16
        perm = np.frombuffer(blob, dtype=pdt, count=C, offset=pos).astype(np.int64)
        pos += C * np.dtype(pdt).itemsize
        if not np.array_equal(np.sort(perm), np.arange(C)):
            raise SystemExit("dq column order is not a permutation")
    ns = 2 if flags & 4 else 4
    tab = zstandard.ZstdDecompressor().decompress(blob[pos : pos + tlen], max_output_size=ncls * ns * 520)
    pos += tlen
    nctx = ncls * ns
    freq = np.zeros((nctx, 256), dtype=np.int64)
    tp = 0
    for c in range(nctx):
        smin, nsym = (int(v) for v in np.frombuffer(tab, dtype=np.int16, count=2, offset=tp))
        tp += 4
        freq[c, smin : smin + nsym] = np.frombuffer(tab, dtype=np.uint16, count=nsym, offset=tp)
        tp += 2 * nsym
    if tp != len(tab):
        raise SystemExit("dq table size mismatch")
    cum = np.zeros((nctx, 257), dtype=np.int64)
    cum[:, 1:] = np.cumsum(freq, axis=1)
    if not np.all((cum[:, 256] == 1 << pb) | (cum[:, 256] == 0)):
        raise SystemExit("dq rANS table does not sum to 4096")
    # per (context, slot): frequency and (slot - cumulative) of the symbol, its output code and the next state
    nslot = 1 << pb
    fs = np.zeros((nctx, nslot), dtype=np.uint64)
    bs = np.zeros((nctx, nslot), dtype=np.uint64)
    nout = np.zeros((nctx, nslot), dtype=np.int8)
    par = np.zeros((nctx, nslot), dtype=np.int64)
    slots = np.arange(nslot, dtype=np.int64)
    for c in range(nctx):
        if not cum[c, 256]:
            continue
        sym = np.repeat(np.arange(256), freq[c])
        k = sym - 128
        q = ((c % ns) >> 1) if ns == 4 else (c % ns)
        fs[c] = freq[c, sym].astype(np.uint64)
        bs[c] = (slots - cum[c, sym]).astype(np.uint64)
        nout[c] = (2 * k - q * np.sign(k)).astype(np.int8)
        par[c] = k & 1
    fs, bs, nout, par = fs.reshape(-1), bs.reshape(-1), nout.reshape(-1), par.reshape(-1)
    _, rcls = np.unique(key, return_inverse=True)
    rcls = rcls.astype(np.int64).reshape(-1)
    if int(rcls.max(initial=0)) + 1 != ncls:
        raise SystemExit("dq class count mismatch")
    seg = m * ch
    n = R * C
    lanes = -(-n // seg)
    x = np.frombuffer(blob, dtype=np.uint32, count=lanes, offset=pos).astype(np.uint64)
    pos += 4 * lanes
    words = np.frombuffer(blob, dtype=np.uint16, offset=pos).astype(np.uint64)
    ptr = 0
    clsb = np.zeros((lanes * seg) // C + 1, dtype=np.int64)
    clsb[:R] = rcls * (ns * nslot)  # flat table offset of each row's first context
    out = np.zeros((lanes, seg), dtype=np.int8)
    base = np.arange(lanes, dtype=np.int64) * seg
    partial = n % seg != 0
    valid = None
    st = np.zeros(lanes, dtype=np.int64)
    sh = 0 if ns == 4 else 1
    mask_slot, s_pb, s16, lowv = np.uint64(nslot - 1), np.uint64(pb), np.uint64(16), np.uint64(low)
    for i in range(seg):
        if i % ch == 0:
            st[:] = 0  # a chunk starts in every lane
        pos_i = base + i
        if partial:
            valid = pos_i < n
        fi = clsb[pos_i // C] + ((st >> sh) << pb) + (x & mask_slot).astype(np.int64)
        xn = fs[fi] * (x >> s_pb) + bs[fi]
        x = np.where(valid, xn, x) if partial else xn
        need = np.nonzero((x < lowv) & valid if partial else x < lowv)[0]
        if len(need):
            x[need] = (x[need] << s16) | words[ptr : ptr + len(need)]
            ptr += len(need)
        out[:, i] = nout[fi]
        st = DQ_NEXT[st, par[fi]]
    if ptr != len(words):
        raise SystemExit("dq stream not fully consumed")
    out = out.reshape(-1)[:n].reshape(R, C)
    if perm is None:
        return out
    nat = np.empty_like(out)
    nat[:, perm] = out
    return nat


def decode_tensor(blobs: Blobs, t: dict, rans_codes: np.ndarray | None = None) -> torch.Tensor:
    shape = t["shape"]
    n = int(np.prod(shape)) if shape else 1
    if t["enc"] == "raw":
        return bf16_from(blobs.get(t["data"], 2 * n), shape).clone()
    if t["enc"] == "q":
        out_f, in_f = shape
        group = t["group"]
        g = in_f // group
        codes = torch.frombuffer(blobs.get(t["codes"], out_f * in_f), dtype=torch.uint8).view(out_f, in_f)
        scale = bf16_from(blobs.get(t["scale"], 2 * out_f * g), (out_f, g))
        zero = torch.frombuffer(blobs.get(t["zero"], out_f * g), dtype=torch.uint8).view(out_f, g)
        return dequant(codes, scale, zero, group)
    if t["enc"] == "u":
        out_f, in_f = shape
        key = np.frombuffer(bytes(blobs.get(t["rowkey"], out_f)), dtype=np.uint8) if "rowkey" in t else None
        if "codes_dq" in t:  # trellis (dependent quantisation) codes, natural row order; contexts from the row keys
            if rans_codes is None:
                rans_codes = dq_decode(blobs.raw(t["codes_dq"]), out_f, in_f, key)
            codes = torch.from_numpy(rans_codes)
            key = None  # natural order already
        elif "codes_rans" in t:  # rANS-coded codes (stored row order; contexts from the row keys)
            sorted_key = np.sort(key, kind="stable") if key is not None else np.zeros(out_f, dtype=np.uint8)
            if rans_codes is None:
                rans_codes = rans_decode(blobs.raw(t["codes_rans"]), out_f, in_f, sorted_key)
            codes = torch.from_numpy(rans_codes)
        else:
            codes = torch.frombuffer(blobs.get(t["codes"], out_f * in_f), dtype=torch.int8).view(out_f, in_f)
        if key is not None:  # rows were stored sorted (stable) by a 1-byte key: put them back
            perm = torch.from_numpy(np.argsort(key, kind="stable").astype(np.int64))
            unsorted = torch.empty_like(codes)
            unsorted[perm] = codes
            codes = unsorted
        step = torch.frombuffer(blobs.get(t["step"], 4 * out_f), dtype=torch.float32)
        col = torch.frombuffer(blobs.get(t["col"], 4 * in_f), dtype=torch.float32) if "col" in t else None
        return dequant_u(codes, step, col)
    raise SystemExit(f"unknown encoding {t['enc']}")


_BLOBS = None


def _rans_codes(t: dict) -> np.ndarray:
    """Worker: rANS-decode one tensor's codes (numpy only; no torch in forked workers)."""
    out_f, in_f = t["shape"]
    key = np.frombuffer(bytes(_BLOBS.get(t["rowkey"], out_f)), dtype=np.uint8)
    if "codes_dq" in t:
        return dq_decode(_BLOBS.raw(t["codes_dq"]), out_f, in_f, key)
    return rans_decode(_BLOBS.raw(t["codes_rans"]), out_f, in_f, np.sort(key, kind="stable"))


def decoded_tensors(blobs: Blobs, tensors: list[dict]):
    """Tensors in manifest order. The rANS stage (numpy, integer-only) runs in a fork pool over tensors; dequantisation
    stays in this process. Results are consumed in manifest order, so the output does not depend on the worker count."""
    global _BLOBS
    _BLOBS = blobs
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    coded = [t for t in tensors if ("codes_rans" in t or "codes_dq" in t) and "rowkey" in t]
    if ncpu > 1 and coded:
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(min(16, ncpu)) as pool:
            it = pool.imap(_rans_codes, coded, chunksize=1)
            for t in tensors:
                pre = next(it) if (coded and t is coded[0]) else None
                if pre is not None:
                    coded.pop(0)
                yield decode_tensor(blobs, t, pre).contiguous()
    else:
        for t in tensors:
            yield decode_tensor(blobs, t).contiguous()


def main(artifact: str, output: str) -> None:
    with open(find_meta(artifact)) as f:
        meta = json.load(f)
    if meta.get("format") != "ourq-ff" or meta.get("version") != 1:
        raise SystemExit("unsupported artifact format")
    blobs = Blobs(artifact, meta["files"])
    shard, shard_bytes, index = {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, index
        if shard:
            save_file(shard, os.path.join(output, f"model-{index:05d}.safetensors"), metadata={"format": "pt"})
            index += 1
            shard, shard_bytes = {}, 0

    for t, w in zip(meta["tensors"], decoded_tensors(blobs, meta["tensors"])):
        if list(w.shape) != list(t["shape"]):
            raise SystemExit(f"{t['name']}: shape {list(w.shape)} != {t['shape']}")
        shard[t["name"]] = w
        shard_bytes += w.numel() * 2
        if shard_bytes >= SHARD_BYTES:
            flush()
    flush()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: decode.py ARTIFACT_DIR OUTPUT_DIR")
    main(sys.argv[1], sys.argv[2])
