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

import torch
import zstandard

from quant import common
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
        return {
            "codes": q.codes.to(torch.uint8).contiguous().cpu().numpy().tobytes(),
            "scale": q.scale.to(torch.bfloat16).contiguous().cpu().view(torch.uint8).numpy().tobytes(),
            "zero": q.zero.to(torch.uint8).contiguous().cpu().numpy().tobytes(),
        }

    def dequant(self) -> torch.Tensor:
        return self.tensor.to(torch.bfloat16) if self.q is None else self.q.dequant(self.group, torch.bfloat16)


def _compress(raw: bytes, level: int) -> bytes:
    return zstandard.ZstdCompressor(level=level, write_content_size=True).compress(raw)


def encode(entries: list[Entry], level: int = 9, workers: int = 16) -> tuple[list[dict], list[bytes]]:
    """-> (tensor records with blob refs into the frame list, frames). Refs are (frame index) until laid out."""
    jobs = []
    records = []
    for e in entries:
        rec = {"name": e.name, "shape": e.shape, "enc": "raw" if e.q is None else "q"}
        if e.q is not None:
            rec["group"], rec["bits"] = e.group, e.bits
        for key, raw in e.blobs().items():
            rec[key] = len(jobs)
            jobs.append(raw)
        records.append(rec)
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(lambda raw: _compress(raw, level), jobs))
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
        for key in ("data", "codes", "scale", "zero"):
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
