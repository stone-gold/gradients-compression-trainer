"""Round trip of 4-state dependent quantisation on real layer tensors (saved by the L5 microtest capture: one decoder
block's weights, input second moments H and per-row Fisher), through the trainer's path: q/k/v and gate/up stacked
like freeform's group quantiser, bucketed act order, optional rate tables, ffcodec.encode(use_rans=True), the real
artifact writer and decoder. Writes the artifact to OUT_DIR/art for the 1-vs-16-thread decode check.
usage: PYTHONPATH=<repo> python tools/test_dq_real.py DATA.pt OUT_DIR [lam]
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

from quant import dq, ffcodec

data = torch.load(sys.argv[1])
out = Path(sys.argv[2])
lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
art = out / "art"
art.mkdir(parents=True, exist_ok=True)
names = sorted(k[2:] for k in data if k.startswith("W_"))
groups = {"H_qkv": [n for n in names if any(p in n for p in ("q_proj", "k_proj", "v_proj"))], "H_o": [n for n in names if "o_proj" in n],
          "H_gateup": [n for n in names if any(p in n for p in ("gate_proj", "up_proj"))], "H_down": [n for n in names if "down_proj" in n]}
res = {}
for hk, ns in groups.items():
    ws = [data["W_" + n] for n in ns]
    steps = []
    for n, w in zip(ns, ws):
        f = data["F_" + n].float()
        ft = float(f.mean())
        base = (1.0 / math.sqrt(ft)) * (ft / f.clamp(min=ft * 1e-6)).sqrt().clamp(0.25, 4.0)
        c0 = 0.2 * float(w.pow(2).mean().sqrt()) / math.exp(float(base.log().mean()))
        st = torch.maximum(base * c0 * 4.0, w.abs().amax(1) / 100.0)
        steps.append(torch.maximum(0.6 * st, w.abs().amax(1) / 100.0))
    wcat, d = torch.cat(ws, 0), torch.cat(steps)
    kw = {}
    if lam > 0:
        bits, cls = dq.rate_model(wcat, d, "cpu", 512)
        kw = {"lam": lam, "bits": bits, "cls": cls}
    t0 = time.time()
    u = dq.gptq_dq(wcat, data[hk].float(), d, act_order="b16", split=[w.shape[0] for w in ws], **kw)
    print(hk, tuple(wcat.shape), "%.1fs" % (time.time() - t0), flush=True)
    r0 = 0
    for i, (n, w) in enumerate(zip(ns, ws)):
        res[n] = dq.UResult(u.codes[r0 : r0 + w.shape[0]].contiguous(), u.step[r0 : r0 + w.shape[0]].contiguous(), dq=True, perm=u.perm, okey=u.okey,
                            chunk=u.chunk, est=u.est[i])
        r0 += w.shape[0]
entries = [ffcodec.Entry(n, list(res[n].codes.shape), q=res[n]) for n in names]
t0 = time.time()
records, frames = ffcodec.encode(entries, level=0, workers=8, use_rans=True)
enc_s = time.time() - t0
total = ffcodec.write_artifact(art, "l5dq-real", records, frames, {"candidate": "dq-real"})
spec = importlib.util.spec_from_file_location("ff_decoder", Path(__file__).resolve().parent.parent / "decoder" / "decode.py")
dec = importlib.util.module_from_spec(spec)
sys.modules["ff_decoder"] = dec
spec.loader.exec_module(dec)
meta = json.loads(Path(dec.find_meta(str(art))).read_text())
blobs = dec.Blobs(str(art), meta["files"])
by = {e.name: e for e in entries}
t0 = time.time()
for t in meta["tensors"]:
    got = dec.decode_tensor(blobs, t)
    assert torch.equal(got.view(torch.int16), by[t["name"]].dequant().contiguous().view(torch.int16)), t["name"]
dec_s = time.time() - t0
coders = {t["name"].split(".")[-2]: [k for k in t if k.startswith("codes")][0] for t in meta["tensors"]}
for t in meta["tensors"]:
    if "codes_dq" in t:
        print("est vs coded", t["name"].split(".")[-2], by[t["name"]].q.est, t["codes_dq"][2], "%+.2f%%" % (100.0 * (by[t["name"]].q.est / t["codes_dq"][2] - 1)))
print(json.dumps({"tensors": len(meta["tensors"]), "bytes": total, "coders": coders, "encode_s": round(enc_s, 1), "decode_s_serial": round(dec_s, 1)}))
print("OK bit-exact")
