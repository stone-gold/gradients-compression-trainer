"""compressed-tensors pack-quantized layout, as the vLLM 0.31 evaluator reads it.

Unsigned codes u in [0, 2^bits - 1] (compressed-tensors stores signed q + 2^(bits-1), which is the same number),
packed as a dense little-endian bitstream along one dimension: value k occupies bits [k*bits, (k+1)*bits) of the
row's stream, word w holds stream bits [32w, 32w+32), bit 0 = LSB. No padding between values (3-bit straddles words).
Dequant: w = (u - z) * scale, z the unsigned zero point packed the same way along the OUTPUT dimension.
"""

import math

import torch


def pack_last_dim(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[R, N] integer codes in [0, 2^bits) -> int32 [R, ceil(N*bits/32)]."""
    codes = codes.to(torch.int64)
    rows, n = codes.shape
    assert int(codes.min()) >= 0 and int(codes.max()) < (1 << bits), "code out of range"
    words = math.ceil(n * bits / 32)
    if 32 % bits == 0:
        per = 32 // bits
        pad = words * per - n
        if pad:
            codes = torch.nn.functional.pad(codes, (0, pad))
        codes = codes.view(rows, words, per)
        shifts = (torch.arange(per, dtype=torch.int64, device=codes.device) * bits).view(1, 1, per)
        packed = (codes << shifts).sum(-1)
    else:
        bit_idx = torch.arange(bits, dtype=torch.int64, device=codes.device)
        stream = ((codes.unsqueeze(-1) >> bit_idx) & 1).to(torch.uint8).view(rows, n * bits)
        pad = words * 32 - n * bits
        if pad:
            stream = torch.nn.functional.pad(stream, (0, pad))
        stream = stream.view(rows, words, 32).to(torch.int64)
        shifts = torch.arange(32, dtype=torch.int64, device=codes.device).view(1, 1, 32)
        packed = (stream << shifts).sum(-1)
    # two's-complement wrap into int32
    packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return packed.to(torch.int32)


def unpack_last_dim(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Inverse of pack_last_dim (for self-checks)."""
    p = packed.to(torch.int64) & 0xFFFFFFFF
    rows, words = p.shape
    shifts = torch.arange(32, dtype=torch.int64, device=p.device)
    stream = ((p.unsqueeze(-1) >> shifts) & 1).view(rows, words * 32)[:, : n * bits]
    stream = stream.view(rows, n, bits)
    weights = (1 << torch.arange(bits, dtype=torch.int64, device=p.device)).view(1, 1, bits)
    return (stream * weights).sum(-1)


def packed_tensors(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None, bits: int) -> dict:
    """Tensors replacing one Linear weight [out, in]: codes [out, in], scale [out, G], zero [out, G] (unsigned)."""
    out_f, in_f = codes.shape
    result = {
        "weight_packed": pack_last_dim(codes, bits),
        "weight_scale": scale.to(torch.bfloat16).contiguous(),
        "weight_shape": torch.tensor([out_f, in_f], dtype=torch.int64),
    }
    if zero is not None:
        # packed along dim 0: transpose to [G, out], pack the last dim, transpose back to [ceil(out*bits/32), G]
        result["weight_zero_point"] = pack_last_dim(zero.t().contiguous(), bits).t().contiguous()
    return result
