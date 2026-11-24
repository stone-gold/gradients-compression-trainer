"""Round-trip tests for 4-state dependent quantisation (quant/dq.py + quant/rans.py encode_dq + decoder/decode.py).

CPU only. Random tensors with odd shapes (incl. 1 x 1, a column count above 65535 for the uint32 column order, and
dead input columns), trellis codes from rtn_dq and gptq_dq (with and without act order and rate tables), written as a
real artifact by quant/ffcodec.py and decoded with the real decoder: dequantised bf16 must be bit-identical.
usage: PYTHONPATH=<repo> python tools/test_dq.py OUT_DIR
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from quant import dq, ecq, ffcodec, rans

torch.manual_seed(0)
np.random.seed(0)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("QUANT_FF_DQ_CODER", "1")

spec = importlib.util.spec_from_file_location("ff_decoder", Path(__file__).resolve().parent.parent / "decoder" / "decode.py")
dec = importlib.util.module_from_spec(spec)
sys.modules["ff_decoder"] = dec  # the decoder's fork pool pickles its worker function by module name
spec.loader.exec_module(dec)


def heavy(R, C, scale=0.02):
    w = torch.randn(R, C) * scale
    w += torch.randn(R, C).pow(3) * scale * 0.2  # heavier tails
    w[:, ::7] *= 3.0
    return w


def check_states(u):
    ct = u.codes if u.perm is None else u.codes[:, u.perm]
    dq.states_and_k(ct, u.chunk)  # raises if not a path
    rans.dq_symbols(ct.numpy(), u.chunk)


cases = []
# (name, R, C, kind)
for name, R, C, kind in [("one", 1, 1, "rtn"), ("tiny", 3, 7, "rtn"), ("odd", 257, 1000, "rtn"), ("wide", 2, 70001, "rtn"),
                         ("g1", 1, 5, "gptq"), ("g_odd", 129, 333, "gptq"), ("g_nat", 64, 300, "gptq_nat"),
                         ("g_dead", 50, 260, "gptq_dead"), ("g_rate", 300, 512, "gptq_rate"), ("g_col", 70, 200, "gptq_col"), ("g_b16", 33, 301, "gptq_b16"), ("g_b16w", 3, 1101, "gptq_b16"), ("g_chunk", 40, 1024, "gptq_b16"), ("r_chunk", 300, 2048, "rtn"),
                         ("big_codes", 40, 300, "rtn_big")]:
    w = heavy(R, C)
    step = (w.abs().amax(1) / 40.0).clamp(min=1e-6)
    if kind == "rtn":
        u = dq.rtn_dq(w, step, device="cpu", blocksize=64)
    elif kind == "rtn_big":
        u = dq.rtn_dq(w, (w.abs().amax(1) / 130.0).clamp(min=1e-6), device="cpu")  # forces clamping at |n| 126/127
    else:
        X = torch.randn(min(4 * C + 8, 3000), C)
        X[:, ::5] *= 4.0
        if kind == "gptq_dead":
            X[:, 3::11] = 0.0
        H = X.t() @ X / X.shape[0]
        kw = {}
        if kind == "gptq_rate":
            u0 = dq.rtn_dq(w, step, device="cpu")
            bits, cls = dq.rate_tables(u0.codes, None)
            kw = {"lam": 0.1, "bits": bits, "cls": cls}
        if kind == "gptq_col":
            kw["col"] = ecq.column_factors(H, 0.5, 4.0)
        u = dq.gptq_dq(w, H, step, blocksize=32, act_order="b16" if kind == "gptq_b16" else (kind != "gptq_nat"), **kw)
    check_states(u)
    cases.append((name, u))
    print(name, tuple(u.codes.shape), "max|n|", int(u.codes.abs().max()), "perm", u.perm is not None, flush=True)

# direct coder round trip on every case (the artifact writer below keeps zstd where it is smaller)
for name, u in cases:
    c = u.codes.numpy()
    key = torch.clamp(torch.round(16.0 * torch.log2(u.codes.float().abs().mean(1) + 1e-3)) + 128, 0, 255).to(torch.uint8).numpy()
    perm = None if u.perm is None else u.perm.numpy()
    okey = None if u.okey is None else u.okey.numpy()
    blob = rans.encode_dq(c, perm, key, okey=okey, chunk=u.chunk)
    back = dec.dq_decode(blob, c.shape[0], c.shape[1], key)
    assert np.array_equal(back, c), name
    est = rans.estimate_dq_bytes(c, perm, key, okey=okey is not None, chunk=u.chunk)
    print(f"direct {name}: {len(blob)} B (ideal estimate {est} B)", flush=True)
entries = [ffcodec.Entry(f"t.{name}.weight", list(u.codes.shape), q=u) for name, u in cases]
entries.append(ffcodec.Entry("t.bias", [5], tensor=torch.randn(5)))
records, frames = ffcodec.encode(entries, level=0, workers=4, use_rans=True)
tags = {r["name"]: [k for k in r if k.startswith("codes")] for r in records}
print("coders:", json.dumps(tags))
total = ffcodec.write_artifact(out, "test", records, frames, {"candidate": "test"})
meta = json.loads(Path(dec.find_meta(str(out))).read_text())
blobs = dec.Blobs(str(out), meta["files"])
by = {e.name: e for e in entries}
n_dq = 0
for t in meta["tensors"]:
    got = dec.decode_tensor(blobs, t)
    want = by[t["name"]].dequant().cpu()
    assert torch.equal(got.view(torch.int16), want.contiguous().view(torch.int16)), t["name"]
    n_dq += "codes_dq" in t
# the pool path (fork workers) as in the shipped decoder
got_pool = list(dec.decoded_tensors(blobs, meta["tensors"]))
for t, g in zip(meta["tensors"], got_pool):
    assert torch.equal(g.view(torch.int16), by[t["name"]].dequant().cpu().contiguous().view(torch.int16)), t["name"]
assert n_dq >= 1, "no tensor took the dq coder"
# a scalar (non-trellis) code tensor flagged dq must fall back to the static coders, not fail
bad = ecq.rtn_u(heavy(64, 256), torch.full((64,), 0.002), device="cpu")
bad.dq = True
r2, f2 = ffcodec.encode([ffcodec.Entry("bad.weight", [64, 256], q=bad)], use_rans=True)
assert "codes_dq" not in r2[0]
# corrupt stream is detected
t0 = next(t for t in meta["tensors"] if "codes_dq" in t)
raw = bytearray(blobs.raw(t0["codes_dq"]))
raw[-3] ^= 0x5A
try:
    key = np.frombuffer(bytes(blobs.get(t0["rowkey"], t0["shape"][0])), dtype=np.uint8)
    res = dec.dq_decode(bytes(raw), t0["shape"][0], t0["shape"][1], key)
    same = np.array_equal(res, by[t0["name"]].q.codes.numpy())
    print("corrupted stream decoded without error; equal to original:", same)
except SystemExit as e:
    print("corrupted stream rejected:", e)
except Exception as e:  # noqa: BLE001
    print("corrupted stream raised", type(e).__name__)
print(f"OK: {len(meta['tensors'])} tensors bit-exact ({n_dq} with codes_dq), artifact {total} bytes")
