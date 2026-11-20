"""Freeform track (QuantFreeformTask): any bytes plus our decoder/ that expands them to the input model's bf16 tensors.

Score (docs/miner_compression.md#freeform): S = total_bytes * exp(curv * max(0, r - knee)^2), entries over the cap rank
behind. So we pick, among candidates, the one with the lowest predicted S, where r is predicted on the calibration rows
(disjoint from train and the scored set) as loss(candidate) / loss(bf16 input), both measured here with the same code.

Pipeline (each candidate replaces the artifact only when its predicted S is lower; a valid manifest is always in place):
  0. W8 g128 RTN on every 2-D weight (r ~ 1.000): an always-valid fallback within minutes.
  1. GPTQ (Frantar et al. 2023, arXiv:2210.17323) on the decoder Linears for each (bits, group) in QUANT_FF_CANDIDATES,
     with Hessians from train rows; embeddings / untied lm_head by clip-searched RTN at each of QUANT_FF_EMBED_BITS.
Encoding: one uint8 code per weight and bf16 group scales / uint8 zeros, each zstd-compressed (zstd's Huffman literal
coder brings codes down to about their empirical entropy, below the nominal bit width).
"""

import gc
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from quant import common
from quant import ffcodec
from quant.data import load_calibration
from quant.data import load_train_rows
from quant.fixed import mean_loss
from quant.fixed import set_weights
from quant.quantize import QResult
from quant.quantize import QSpec
from quant.quantize import quantize_model_gptq
from quant.quantize import rtn


log = logging.getLogger("quant.freeform")
EMBED_MARKERS = ("embed_tokens", "wte", "word_embeddings")
LM_HEAD = "lm_head.weight"
HF_GITATTRIBUTES_BYTES = 1519  # the .gitattributes Hugging Face adds to a new repo; counted in total_bytes


class BaseTensors:
    """name -> (file, shape) over every *.safetensors in the input dir, as the validator's output check enumerates."""

    def __init__(self, path: str):
        self.where: dict[str, Path] = {}
        self.shapes: dict[str, list[int]] = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as sf:
                for name in sf.keys():
                    self.where[name] = f
                    self.shapes[name] = list(sf.get_slice(name).get_shape())

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.where[name]), framework="pt") as sf:
            return sf.get_tensor(name)

    def bf16_bytes(self) -> int:
        return sum(math.prod(s) * 2 for s in self.shapes.values())


def is_embed_like(name: str) -> bool:
    return name == LM_HEAD or any(m in name for m in EMBED_MARKERS)


def penalised(total_bytes: int, r: float, target: dict) -> float:
    knee, cap, curv = target["knee"], target["cap"], target["penalty_curvature"]
    if r > cap:
        return float("inf")
    return total_bytes * math.exp(curv * max(0.0, r - knee) ** 2)


def freeform_target() -> dict:
    t = common.quant_target()
    return {
        "knee": float(t.get("knee", 1.04)),
        "cap": float(t.get("cap", 1.08)),
        "penalty_curvature": float(t.get("penalty_curvature", 2500.0)),
    }


def decoder_xz_estimate() -> int:
    """Counted decoder bytes (git archive | xz). A constant is close enough for choosing between candidates."""
    return int(os.environ.get("QUANT_FF_DECODER_BYTES", 3000))


def build_entries(base: BaseTensors, linear: dict[str, QResult], lin_group: int, lin_bits: int, embed: dict[str, QResult], emb_group: int, emb_bits: int) -> list[ffcodec.Entry]:
    entries = []
    for name, shape in base.shapes.items():
        if name in linear:
            entries.append(ffcodec.Entry(name, shape, q=linear[name], group=lin_group, bits=lin_bits))
        elif name in embed:
            entries.append(ffcodec.Entry(name, shape, q=embed[name], group=emb_group, bits=emb_bits))
        else:
            entries.append(ffcodec.Entry(name, shape, tensor=base.tensor(name)))
    return entries


def verify_roundtrip(out: Path, entries: list[ffcodec.Entry], sample: int | None = None) -> None:
    """Decode the written artifact with the real decoder module and check bit-equality with what we evaluated."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "decoder" / "decode.py"
    spec = importlib.util.spec_from_file_location("ff_decoder", path)
    dec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dec)
    meta = json.loads(Path(dec.find_meta(str(out))).read_text())
    blobs = dec.Blobs(str(out), meta["files"])
    by_name = {e.name: e for e in entries}
    records = meta["tensors"] if sample is None else meta["tensors"][:: max(1, len(meta["tensors"]) // sample)]
    for t in records:
        got = dec.decode_tensor(blobs, t)
        want = by_name[t["name"]].dequant().cpu()
        if not torch.equal(got.view(torch.int16), want.contiguous().view(torch.int16)):
            raise RuntimeError(f"decoder round trip differs on {t['name']}")
    log.info("decoder round trip bit-exact on %d tensors", len(records))


def run(args) -> None:
    deadline = common.Deadline.from_hours(args.hours_to_complete)
    target = freeform_target()
    out = common.output_dir(args.task_id, args.expected_repo_name)
    src = common.model_dir(args.model)
    base = BaseTensors(src)
    margin = float(os.environ.get("QUANT_FF_MARGIN", 0.006))
    level = int(os.environ.get("QUANT_FF_ZSTD", 9))
    log.info("freeform task %s: model %s (%.2f GB bf16), target %s, budget %.0fs", args.task_id, src, base.bf16_bytes() / 1e9, target, deadline.left())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names_2d = [n for n, s in base.shapes.items() if len(s) == 2 and n.endswith(".weight")]

    def group_for(shape, g):
        return g if shape[1] % g == 0 else None

    best = {"S": float("inf"), "name": None, "r": None, "bytes": None}

    def consider(name: str, entries: list[ffcodec.Entry], r_pred: float, force: bool = False) -> None:
        records, frames = ffcodec.encode(entries, level=level)
        est = ffcodec.encoded_size(frames) + 2048 + HF_GITATTRIBUTES_BYTES + decoder_xz_estimate()
        S = penalised(est, r_pred, target)
        log.info("candidate %s: ~%.1f MB, predicted r %.4f, predicted S %.1f MB (best %s %.1f MB)", name, est / 1e6, r_pred, S / 1e6, best["name"], best["S"] / 1e6)
        if (force or S < best["S"]) and deadline.left() > 60:
            info = {"candidate": name, "predicted_r": r_pred}
            total = ffcodec.write_artifact(out, args.task_id, records, frames, info)
            best.update(S=S, name=name, r=r_pred, bytes=total)
            if os.environ.get("QUANT_FF_VERIFY", "1") == "1":
                verify_roundtrip(out, entries, sample=None if base.bf16_bytes() < 4e9 else 24)

    # 0. fallback: W8 g128 RTN on every 2-D weight
    t0 = time.time()
    q8 = QSpec(bits=8, group=128)
    fallback = {n: rtn(base.tensor(n), q8, device=device) for n in names_2d if group_for(base.shapes[n], 128)}
    consider("rtn-w8g128", build_entries(base, fallback, 128, 8, {}, 128, 8), 1.0 + margin, force=True)
    log.info("fallback written in %.0fs", time.time() - t0)
    del fallback
    gc.collect()

    # 1. data and model
    tok = common.load_tokenizer(src)
    calib, _ = load_calibration(os.environ.get("QUANT_CALIBRATION_PATH"))
    dataset_type = json.loads(args.dataset_type or "{}")
    max_len = int(os.environ.get("QUANT_MAX_LEN", 2048))
    train = load_train_rows(args.task_id, args.dataset, dataset_type, tok, max_len, int(os.environ.get("QUANT_TRAIN_ROWS", 2000)))
    n_eval = int(os.environ.get("QUANT_EVAL_ROWS", 400))
    eval_rows = [r for r in calib if r.n_scored > 0][:n_eval]
    if not eval_rows:  # no calibration: hold out train rows
        eval_rows, train = train[-min(200, len(train) // 4) :], train[: -min(200, len(train) // 4)]
    model = common.load_model(src, device_map=device)
    with torch.no_grad():
        ref_loss = mean_loss(model, eval_rows)
    log.info("bf16 loss on %d eval rows: %.5f", len(eval_rows), ref_loss)
    pool = train if len(train) >= 128 else calib + train
    samples = [torch.tensor(r.input_ids) for r in pool[: int(os.environ.get("QUANT_GPTQ_ROWS", 256))]]

    def restore():
        params = dict(model.named_parameters())
        for n in names_2d:
            if n in params:
                params[n].data.copy_(base.tensor(n).to(params[n].dtype).to(params[n].device))

    embed_names = [n for n in names_2d if is_embed_like(n)]
    cands = [tuple(int(x) for x in c.split(":")) for c in os.environ.get("QUANT_FF_CANDIDATES", "4:128,3:128").split(",") if c]
    embed_bits = [int(b) for b in os.environ.get("QUANT_FF_EMBED_BITS", "8,6,4").split(",") if b]
    emb_group = 128
    embed_q = {b: {n: rtn(base.tensor(n), QSpec(bits=b, group=emb_group), device=device) for n in embed_names if group_for(base.shapes[n], emb_group)} for b in embed_bits}

    for bits, group in cands:
        # rough time check: GPTQ ~ a few minutes per B params on an H100; leave room for encode + write
        if deadline.left() < 600:
            log.warning("out of time before candidate W%d g%d", bits, group)
            break
        restore()
        qs = QSpec(bits=bits, group=group)
        t0 = time.time()
        lin = quantize_model_gptq(model, samples, qs, damp=float(os.environ.get("QUANT_DAMP", 0.01)), true_sequential=True, time_left=lambda: deadline.left() - 600)
        lin = {n: q for n, q in lin.items() if n in base.shapes and not is_embed_like(n)}
        # Linears GPTQ skipped (in_features not divisible by group) get RTN when possible, else stay bf16.
        for n in names_2d:
            if n not in lin and not is_embed_like(n) and group_for(base.shapes[n], group):
                lin[n] = rtn(base.tensor(n), qs, device=device)
        set_weights(model, lin, group)
        log.info("GPTQ W%d g%d: %d linears in %.0fs", bits, group, len(lin), time.time() - t0)
        for eb in embed_bits:
            restore_emb = {n: base.tensor(n) for n in embed_names}
            set_weights(model, embed_q[eb], emb_group)
            with torch.no_grad():
                loss = mean_loss(model, eval_rows)
            r = loss / ref_loss
            log.info("W%dg%d emb%d: eval loss %.5f r %.4f", bits, group, eb, loss, r)
            consider(f"gptq-w{bits}g{group}-emb{eb}", build_entries(base, lin, group, bits, embed_q[eb], emb_group, eb), r + margin)
            params = dict(model.named_parameters())
            for n, t in restore_emb.items():
                if n in params:
                    params[n].data.copy_(t.to(params[n].dtype).to(params[n].device))
            if deadline.left() < 300:
                break
    log.info("final: %s, %s bytes, predicted r %s, predicted S %.1f MB, %.0fs elapsed", best["name"], best["bytes"], best["r"], best["S"] / 1e6, deadline.elapsed())
    del model
    gc.collect()
