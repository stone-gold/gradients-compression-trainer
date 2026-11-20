"""Freeform artifact writer, the inverse of decoder/decode.py (see its docstring for the format).

Writes are crash-safe: each artifact generation uses new file names (p-g<N>-<k>.bin, ff-g<N>.json) and the manifest
is replaced atomically last, so a valid manifest is in place at every instant once the first one exists; the previous
generation's files are deleted only after the new manifest is live.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import zstandard

from quant import common
from quant.ecq import UResult
from quant.quantize import QResult


log = logging.getLogger("quant.ffcodec")
PAYLOAD_BYTES = 2 * 1024**3


class Entry:
    """One output tensor: raw bf16 (tensor) or quantised (QResult + group)."""

    def __init__(self, name: str, shape, tensor: torch.Tensor | None = None, q: QResult | None = None, group: int = 0, bits: int = 0):
        self.name, self.shape, self.tensor, self.q, self.group, self.bits = name, list(shape), tensor, q, group, bits

    def blobs(self) -> dict[str, bytes]:
        if self.q is None:
            return {"data": self.tensor.to(torch.bfloat16).contiguous().cpu().view(torch.uint8).numpy().tobytes()}
        q = self.q
        if isinstance(q, UResult):
            codes = q.codes.to(torch.int8).contiguous().cpu().numpy()
            out = {"step": q.step.to(torch.float32).contiguous().cpu().numpy().tobytes()}
            if q.col is not None:
                out["col"] = q.col.to(torch.float32).contiguous().cpu().numpy().tobytes()
            if os.environ.get("QUANT_FF_ROWSORT", "1") == "1" and codes.shape[0] > 1:
                # rows stored in order of a 1-byte spread key (stable sort), so zstd's per-block Huffman tables see rows
                # of similar code entropy together (-2.3% on t05); the decoder rebuilds the order from the key bytes
                spread = np.abs(codes.astype(np.float32)).mean(1)
                key = np.clip(np.round(16.0 * np.log2(spread + 1e-3)) + 128, 0, 255).astype(np.uint8)
                perm = np.argsort(key, kind="stable")
                out["rowkey"] = key.tobytes()
                codes = np.ascontiguousarray(codes[perm])
            out["codes"] = codes.tobytes()
            return out
        return {
            "codes": q.codes.to(torch.uint8).contiguous().cpu().numpy().tobytes(),
            "scale": q.scale.to(torch.bfloat16).contiguous().cpu().view(torch.uint8).numpy().tobytes(),
            "zero": q.zero.to(torch.uint8).contiguous().cpu().numpy().tobytes(),
        }

    def dequant(self) -> torch.Tensor:
        return self.tensor.to(torch.bfloat16) if self.q is None else self.q.dequant(self.group, torch.bfloat16)


def _params(level: int):
    """level >= 1: a plain zstd level. level <= 0 (default): Huffman-only literals, no LZ matching (min_match 7, tiny
    search). On quantised weight codes LZ matches are spurious and cost ~10% over the order-0 entropy at levels 3-15;
    this setting lands within ~1.5% of the empirical entropy, ~30x faster than level 9. Output is a standard zstd
    frame (the decoder is unchanged)."""
    if level >= 1:
        return zstandard.ZstdCompressionParameters.from_level(level, write_content_size=1)
    return zstandard.ZstdCompressionParameters.from_level(
        1, strategy=zstandard.STRATEGY_FAST, min_match=7, search_log=1, hash_log=6, write_content_size=1
    )


def _compress(raw: bytes, level: int, codes: bool) -> bytes:
    """Codes: one frame at `level`. Other blobs (bf16 tensors, scales, zeros): also try level 9, keep the smaller."""
    out = zstandard.ZstdCompressor(compression_params=_params(level)).compress(raw)
    if level <= 0 and not codes:
        alt = zstandard.ZstdCompressor(compression_params=_params(9)).compress(raw)
        if len(alt) < len(out):
            out = alt
    return out


def encode(entries: list[Entry], level: int = 0, workers: int = 16) -> tuple[list[dict], list[bytes]]:
    """-> (tensor records with blob refs into the frame list, frames). Refs are (frame index) until laid out."""
    jobs = []
    records = []
    for e in entries:
        rec = {"name": e.name, "shape": e.shape, "enc": "raw" if e.q is None else "u" if isinstance(e.q, UResult) else "q"}
        if rec["enc"] == "q":
            rec["group"], rec["bits"] = e.group, e.bits
        for key, raw in e.blobs().items():
            rec[key] = len(jobs)
            jobs.append((raw, key == "codes"))
        records.append(rec)
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(lambda job: _compress(job[0], level, job[1]), jobs))
    return records, frames


def encoded_size(frames: list[bytes]) -> int:
    return sum(len(f) for f in frames)


def _next_generation(out: Path) -> int:
    gens = [int(m.group(1)) for p in out.iterdir() if (m := re.fullmatch(r"ff-g(\d+)\.json", p.name))]
    return max(gens, default=0) + 1


def write_artifact(out: Path, task_id: str, records: list[dict], frames: list[bytes], info: dict | None = None) -> int:
    """Lay frames out in payload files, write ff-g<N>.json and the manifest; returns total bytes listed (+manifest)."""
    t0 = time.time()
    gen = _next_generation(out)
    files: list[str] = []
    where: list[tuple[int, int, int]] = []
    fh, cur, off = None, None, 0
    try:
        for frame in frames:
            if fh is None or (off + len(frame) > PAYLOAD_BYTES and off > 0):
                if fh is not None:
                    fh.flush()
                    os.fsync(fh.fileno())
                    fh.close()
                cur = f"p-g{gen}-{len(files)}.bin"
                files.append(cur)
                fh = open(out / cur, "wb")
                off = 0
            fh.write(frame)
            where.append((len(files) - 1, off, len(frame)))
            off += len(frame)
    finally:
        if fh is not None:
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
    tensors = []
    for rec in records:
        r = dict(rec)
        for key in ("data", "codes", "scale", "zero", "step", "rowkey", "col"):
            if key in r:
                r[key] = list(where[r[key]])
        tensors.append(r)
    meta_name = f"ff-g{gen}.json"
    meta = {"format": "ourq-ff", "version": 1, "files": files, "tensors": tensors}
    if info:
        meta["info"] = info
    common.atomic_write_bytes(out / meta_name, json.dumps(meta, separators=(",", ":")).encode())
    listed = files + [meta_name]
    common.write_manifest(out, task_id, "freeform", listed)
    keep = set(listed) | {common.MANIFEST}
    for p in out.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink()
    total = sum((out / n).stat().st_size for n in listed) + (out / common.MANIFEST).stat().st_size
    log.info("freeform artifact g%d written in %.0fs: %d files, %.1f MB", gen, time.time() - t0, len(listed), total / 1e6)
    return total
