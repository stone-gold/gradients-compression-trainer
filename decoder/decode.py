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
Output: top-level *.safetensors with exactly the tensors listed, all bf16. Deterministic: no randomness, no threads
racing on shared output (rANS artifacts decode tensors in a process pool; results are consumed in manifest order).
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


def decode_tensor(blobs: Blobs, t: dict) -> torch.Tensor:
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
        if "codes_rans" in t:  # rANS-coded codes (stored row order; contexts from the row keys)
            sorted_key = np.sort(key, kind="stable") if key is not None else np.zeros(out_f, dtype=np.uint8)
            codes = torch.from_numpy(rans_decode(blobs.raw(t["codes_rans"]), out_f, in_f, sorted_key))
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


def _worker_init() -> None:
    torch.set_num_threads(1)


def _decode_one(t: dict) -> torch.Tensor:
    return decode_tensor(_BLOBS, t).contiguous()


def decoded_tensors(blobs: Blobs, tensors: list[dict]):
    """Tensors in manifest order. rANS-coded artifacts are decoded by a fork pool over tensors (one tensor per task,
    results consumed in order: the output does not depend on the number of workers)."""
    global _BLOBS
    _BLOBS = blobs
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    if ncpu > 1 and any("codes_rans" in t for t in tensors):
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(min(16, ncpu), initializer=_worker_init) as pool:
            yield from pool.imap(_decode_one, tensors, chunksize=1)
    else:
        for t in tensors:
            yield _decode_one(t)


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
