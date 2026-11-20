"""Freeform decoder: python decode.py /artifact /output

Artifact (written by quant/ffcodec.py): ff-g<N>.json (highest N wins) + payload files it names. Every blob is one
zstd frame. Tensor encodings:
  raw: little-endian bf16 bytes of the tensor.
  q:   asymmetric group-wise integer quantisation along the last dim. codes uint8 [out, in] (one code per byte; zstd's
       entropy coder does the bit packing), scale bf16 [out, in/group], zero uint8 [out, in/group];
       w = bf16((code - zero) * scale), computed in float32 (bit-identical to the trainer's dequant).
Output: top-level *.safetensors with exactly the tensors listed, all bf16. Deterministic: no randomness, no threads
racing on shared output.
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
    raise SystemExit(f"unknown encoding {t['enc']}")


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

    for t in meta["tensors"]:
        w = decode_tensor(blobs, t).contiguous()
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
