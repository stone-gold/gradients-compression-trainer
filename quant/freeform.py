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

import numpy as np
import torch
from safetensors import safe_open

from quant import common
from quant import dq
from quant import ecq
from quant import ffcodec
from quant import landing
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
SPECIAL_COUNT = 1 << 30  # train-split count given to special / added tokens (always "frequent")


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
        over = getattr(self, "overrides", {}).get(name)
        if over is not None:  # vocab trim "mean": unused rows replaced before quantisation
            return over.clone()
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
        # QUANT_TARGET is QuantTarget.model_dump_json(): unset fields arrive as null, not missing
        "knee": float(t.get("knee") or 1.04),
        "cap": float(t.get("cap") or 1.08),
        "penalty_curvature": float(t.get("penalty_curvature") if t.get("penalty_curvature") is not None else 2500.0),
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


def row_losses(model, rows) -> list[float]:
    """Per-row mean supervised-token loss (NaN for a row with no scored token after the shift); the mean over the finite
    entries is fixed.mean_loss (the quant_loss convention)."""
    from quant.fixed import IGNORE_INDEX

    device = next(model.parameters()).device
    out = []
    for row in rows:
        ids = torch.tensor(row.input_ids, device=device).view(1, -1)
        labels = torch.tensor(row.labels, device=device)
        target = labels[1:]
        mask = target != IGNORE_INDEX
        if not mask.any():
            out.append(float("nan"))
            continue
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        lp = torch.log_softmax(logits[mask], dim=-1)
        out.append(float(-lp.gather(1, target[mask].view(-1, 1)).mean()))
    return out


def _finite_mean(xs: list[float]) -> float:
    v = [x for x in xs if x == x]
    return sum(v) / max(1, len(v))


def split_encoded(records: list[dict], frames: list[bytes]) -> dict[str, tuple[dict, list[bytes]]]:
    """name -> (record with blob refs local to its own frame list, frames): lets candidates be mixed tensor by tensor."""
    out = {}
    for rec in records:
        r, fl = {}, []
        for k, v in rec.items():
            if k in ("name", "shape", "enc", "group", "bits"):
                r[k] = v
            else:
                r[k] = len(fl)
                fl.append(frames[v])
        out[rec["name"]] = (r, fl)
    return out


def join_encoded(parts: list[tuple[dict, list[bytes]]]) -> tuple[list[dict], list[bytes]]:
    records, frames = [], []
    for rec, fl in parts:
        r = {k: (v + len(frames) if k not in ("name", "shape", "enc", "group", "bits") else v) for k, v in rec.items()}
        records.append(r)
        frames.extend(fl)
    return records, frames


def unit_of(name: str) -> int:
    """Mixing unit: decoder block index, -1 for everything outside the blocks (embeddings, untied head)."""
    import re

    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def tune_candidate(model, lin: dict, train_rows, sel_rows, budget_s: float, pad_id: int, deadline_left) -> dict[str, torch.Tensor]:
    """End-to-end tuning of one ECQ candidate with its integer codes frozen (EfficientQAT E2E-QP, Chen et al. 2024,
    arXiv:2407.11062; the fixed track's quant/tune.py): per-row steps (as one-group-per-row scales) plus norm weights and
    biases, by the evaluator's loss on train rows; the best state by loss on sel_rows (held out of training) is kept.
    Rewrites lin's steps in place (bf16-representable, so the decoder's bf16(code * step) equals the trained forward) and
    returns the tuned raw 1-D tensors {name: bf16}. The model is left with QLinear modules (only measure() after this)."""
    from quant import tune

    qres = {n: QResult(q.codes, q.step.view(-1, 1).float(), torch.zeros(q.codes.shape[0], 1, dtype=torch.uint8)) for n, q in lin.items() if not is_embed_like(n)}
    qlinears = {}
    for name, module in list(model.named_modules()):
        key = name + ".weight"
        if isinstance(module, torch.nn.Linear) and key in qres:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            q = tune.QLinear(module, qres[key], module.weight.shape[1])
            setattr(parent, child, q)
            qlinears[key] = q
            module.weight = None
    torch.cuda.empty_cache()

    def sel_loss() -> float:
        # norms train in fp32 (train_scales): autocast keeps the bf16 head / embeddings consistent during selection
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return _finite_mean(row_losses(model, sel_rows))

    tune.train_scales(
        model, qlinears, train_rows, sel_loss, deadline_left, pad_id,
        lr=float(os.environ.get("QUANT_FF_TUNE_LR", 5e-4)), train_norms=os.environ.get("QUANT_FF_TUNE_NORMS", "1") == "1",
        max_tokens=int(os.environ.get("QUANT_FF_TUNE_TOKENS", 16384)), epochs=float(os.environ.get("QUANT_FF_TUNE_EPOCHS", 3.0)),
        budget_seconds=budget_s, eval_every=int(os.environ.get("QUANT_FF_TUNE_EVAL_EVERY", 50)),
    )
    for key, q in qlinears.items():
        lin[key] = ecq.UResult(lin[key].codes, q.scale().detach().float().view(-1).cpu())
    raw = {}
    for name, p in model.named_parameters():
        if ("norm" in name or name.endswith(".bias")) and p.dim() == 1:
            p.data = p.data.to(torch.bfloat16)
            raw[name] = p.data.detach().cpu().clone()
    return raw


def used_token_ids(task_id: str, dataset_arg: str, tok, extra_rows, time_limit: float) -> dict[int, int] | None:
    """Token ids that occur anywhere in the task's train split (every string field of every row, tokenised in batches by
    the fast tokenizer), in the calibration rows and in the tokenizer's special / added tokens. None if the split cannot
    be read completely within time_limit (then nothing is trimmed). Returned as {id: occurrences in the train split};
    special / added tokens count as frequent (SPECIAL_COUNT)."""
    import collections
    import itertools

    t_end = time.time() + time_limit
    path = next((p for p in (f"/cache/datasets/{task_id}_train_data.json", dataset_arg) if p and os.path.exists(p)), None)
    if path is None:
        return None
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = next((v for v in rows.values() if isinstance(v, list)), [])

    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)

    used: collections.Counter = collections.Counter()
    it = iter(rows)
    while True:
        chunk = list(itertools.islice(it, 512))
        if not chunk:
            break
        if time.time() > t_end:
            log.warning("vocab trim: token scan out of time, no trimming")
            return None
        texts = [t for r in chunk for t in strings(r) if t]
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            used.update(ids)
    for r in extra_rows:
        used.update(r.input_ids)
    special = {int(i) for i in (tok.all_special_ids or [])} | {int(i) for i in getattr(tok, "added_tokens_decoder", {}).keys()}
    for t in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tok, t, None)
        if v is not None:
            special.add(int(v))
    for i in special:
        used[i] = max(used[i], SPECIAL_COUNT)
    return dict(used)


@torch.no_grad()
def head_hessian(model, rows, max_rows: int = 256) -> torch.Tensor | None:
    """E[h h^T] (fp32 [d, d] on the model's device) over the inputs of the output embedding (LM head) at the scored
    positions of `rows`: the metric of a head row's quantisation error d is d^T H d (the variance of its logit error)."""
    from quant.fixed import IGNORE_INDEX

    head = model.get_output_embeddings()
    if head is None:
        return None
    device = next(model.parameters()).device
    got = []
    handle = head.register_forward_pre_hook(lambda _m, inp: got.append(inp[0].detach()))
    H, n = None, 0
    try:
        for row in rows[:max_rows]:
            ids = torch.tensor(row.input_ids, device=device).view(1, -1)
            mask = torch.tensor(row.labels, device=device)[1:] != IGNORE_INDEX
            if not bool(mask.any()):
                continue
            got.clear()
            model(input_ids=ids, use_cache=False)
            if not got:
                continue
            x = got[-1].reshape(-1, got[-1].shape[-1])[:-1][mask].double()
            H = x.t().matmul(x) if H is None else H.add_(x.t().matmul(x))
            n += x.shape[0]
    finally:
        handle.remove()
    if H is None or n < 2 * H.shape[0]:
        return None
    return (H / n).float()


def final_rans(args, out, best: dict, deadline, level: int, base) -> None:
    """Lossless last step (QUANT_FF_RANS=1, default): re-code the shipped candidate's codes with row-class context rANS
    (quant/rans.py; t23: -3.3% on the linears' codes, embeddings keep zstd where LZ wins), verify with the real decoder on
    a sample of tensors, else put the zstd generation back. The weights, and so r, are unchanged."""
    entries = best.get("entries")
    if os.environ.get("QUANT_FF_RANS", "1") != "1" or not entries or str(best.get("name")).startswith("rtn-"):
        return
    n_codes = sum(math.prod(e.shape) for e in entries if e.q is not None)
    need = n_codes / float(os.environ.get("QUANT_FF_RANS_RATE", 2e7)) + 120
    # trellis codes in plain zstd frames cost ~+34% (t23 layer 13): their state-aware coding is not optional, so it
    # runs whenever there is any time left (run_ecq reserves its time when DQ is on)
    has_dq = any(getattr(e.q, "dq", False) for e in entries if e.q is not None)
    if deadline.left() < (60 if has_dq else need + 120):
        log.info("rANS: skipped (%.0fs left, needs ~%.0fs)", deadline.left(), need)
        return
    t0 = time.time()
    old_bytes = best["bytes"]
    try:
        records, frames = ffcodec.encode(entries, level=level, use_rans=True)
        info = dict(best.get("info") or {}, coder="rans")
        total = ffcodec.write_artifact(out, args.task_id, records, frames, info)
        verify_roundtrip(out, entries, sample=int(os.environ.get("QUANT_FF_RANS_VERIFY", 32)))
        best["bytes"] = total
        log.info("rANS: %d -> %d bytes (%+.2f%%) in %.0fs", old_bytes, total, 100.0 * (total / old_bytes - 1), time.time() - t0)
    except Exception as exc:  # noqa: BLE001 - the zstd generation goes back in place
        log.warning("rANS stage failed (%s): restoring the zstd artifact", exc)
        records, frames = ffcodec.encode(entries, level=level)
        best["bytes"] = ffcodec.write_artifact(out, args.task_id, records, frames, best.get("info"))


def run(args) -> None:
    deadline = common.Deadline.from_hours(args.hours_to_complete)
    target = freeform_target()
    out = common.output_dir(args.task_id, args.expected_repo_name)
    src = common.model_dir(args.model)
    base = BaseTensors(src)
    margin = float(os.environ.get("QUANT_FF_MARGIN", 0.006))
    level = int(os.environ.get("QUANT_FF_ZSTD", 0))  # 0 = Huffman-only frames (ffcodec._params)
    log.info("freeform task %s: model %s (%.2f GB bf16), target %s, budget %.0fs", args.task_id, src, base.bf16_bytes() / 1e9, target, deadline.left())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names_2d = [n for n, s in base.shapes.items() if len(s) == 2 and n.endswith(".weight")]

    def group_for(shape, g):
        return g if shape[1] % g == 0 else None

    best = {"S": float("inf"), "name": None, "r": None, "bytes": None, "score": float("inf")}
    # QUANT_FF_LAND=expected (default): rank candidates by expected ln S under the noise model of quant/landing.py and
    # aim at knee + delta; "under": the 17502ba rule (r_cal + margin must stay under the knee).
    land = os.environ.get("QUANT_FF_LAND", "expected")
    tgt = landing.Target.from_quant_target(common.quant_target())
    log.info("landing %s: knee r %.4f cap r %.4f curvature %.1f (mode %s)", land, tgt.knee_r, tgt.cap_r, tgt.curv, landing.TARGET_MODE)

    def consider(name: str, entries: list[ffcodec.Entry] | None, r_pred: float, force: bool = False, sigma: float | None = None, enc=None):
        records, frames = enc if enc is not None else ffcodec.encode(entries, level=level)
        est = ffcodec.encoded_size(frames) + 2048 + HF_GITATTRIBUTES_BYTES + decoder_xz_estimate()
        if entries is not None:
            # trellis codes sit in plain zstd frames until the final coding (+~34%): count their estimated final size
            dq_est = {e.name: e.q.est for e in entries if e.q is not None and getattr(e.q, "dq", False) and isinstance(getattr(e.q, "est", None), int)}
            for rec in records:
                if rec["name"] in dq_est and "codes" in rec:
                    est += dq_est[rec["name"]] - len(frames[rec["codes"]])
        if land == "expected":
            sg = sigma if sigma is not None else 0.003
            score = tgt.expected_ln_s(est, r_pred, sg)
            S = math.exp(min(score, 700.0)) if score < float("inf") else float("inf")
            log.info("candidate %s: ~%.1f MB, predicted r %.4f +- %.4f, P(r>knee) %.2f, expected S %.1f MB (best %s %.1f MB)", name, est / 1e6,
                     r_pred, sg, tgt.p_over(r_pred, sg, tgt.knee_r), S / 1e6, best["name"], best["S"] / 1e6)
            if (force or score < best["score"]) and deadline.left() > 60:
                info = {"candidate": name, "predicted_r": r_pred, "sigma": sg}
                total = ffcodec.write_artifact(out, args.task_id, records, frames, info)
                best.update(S=S, score=score, name=name, r=r_pred, bytes=total, entries=entries, info=info)
                if os.environ.get("QUANT_FF_VERIFY", "1") == "1" and entries is not None:
                    verify_roundtrip(out, entries, sample=None if base.bf16_bytes() < 4e9 else 24)
            return records, frames, est
        S = penalised(est, r_pred, target)
        log.info("candidate %s: ~%.1f MB, predicted r %.4f, predicted S %.1f MB (best %s %.1f MB)", name, est / 1e6, r_pred, S / 1e6, best["name"], best["S"] / 1e6)
        # r_pred = measured calibration r + margin. A candidate predicted over the knee never replaces one predicted under
        # it (the small predicted penalty is not worth the risk of a real one); one under the knee always beats one over.
        feasible = r_pred <= target["knee"]
        # (the W8 fallback does not count: a slightly penalised 4-bit result is still far smaller than it)
        best_feasible = best["r"] is not None and best["r"] <= target["knee"] and not str(best["name"]).startswith("rtn-")
        if not force and best_feasible and not feasible:
            log.info("candidate %s predicted over the knee: kept %s", name, best["name"])
            return records, frames, est
        better = S < best["S"] or (feasible and not best_feasible)
        if (force or better) and deadline.left() > 60:
            info = {"candidate": name, "predicted_r": r_pred}
            total = ffcodec.write_artifact(out, args.task_id, records, frames, info)
            best.update(S=S, name=name, r=r_pred, bytes=total)
            if os.environ.get("QUANT_FF_VERIFY", "1") == "1" and entries is not None:
                verify_roundtrip(out, entries, sample=None if base.bf16_bytes() < 4e9 else 24)
        return records, frames, est

    # 0. fallback: W8 g128 RTN on every 2-D weight
    t0 = time.time()
    fb_bits = int(os.environ.get("QUANT_FF_FALLBACK_BITS", 8))
    q8 = QSpec(bits=fb_bits, group=128)
    fallback = {n: rtn(base.tensor(n), q8, device=device) for n in names_2d if group_for(base.shapes[n], 128)}
    consider(f"rtn-w{fb_bits}g128", build_entries(base, fallback, 128, fb_bits, {}, 128, fb_bits), 1.0 + margin, force=True)
    log.info("fallback written in %.0fs", time.time() - t0)
    del fallback
    gc.collect()

    if os.environ.get("QUANT_FF_MODE", "ecq") == "ecq":
        run_ecq(args, deadline, target, base, names_2d, consider, best, margin, device, src, tgt if land == "expected" else None)
        final_rans(args, out, best, deadline, level, base)
        log.info("final: %s, %s bytes, predicted r %s, predicted S %.1f MB, %.0fs elapsed", best["name"], best["bytes"], best["r"], best["S"] / 1e6, deadline.elapsed())
        return

    raw_cands = os.environ.get("QUANT_FF_CANDIDATES", "auto")
    auto = raw_cands == "auto"
    # auto: start at W4 g128; above the knee step up a bit (to W8), comfortably under it try one bit fewer (to W3).
    cands = [(4, 128)] if auto else [tuple(int(x) for x in c.split(":")) for c in raw_cands.split(",") if c]
    if not cands:
        log.info("no GPTQ candidates: done with %s", best["name"])
        return

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
    embed_bits = [int(b) for b in os.environ.get("QUANT_FF_EMBED_BITS", "8,6,4").split(",") if b]
    emb_group = 128
    embed_q = {b: {n: rtn(base.tensor(n), QSpec(bits=b, group=emb_group), device=device) for n in embed_names if group_for(base.shapes[n], emb_group)} for b in embed_bits}

    tried = set()
    while cands:
        bits, group = cands.pop(0)
        tried.add((bits, group))
        r_min = float("inf")
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
            r_min = min(r_min, r)
            log.info("W%dg%d emb%d: eval loss %.5f r %.4f", bits, group, eb, loss, r)
            consider(f"gptq-w{bits}g{group}-emb{eb}", build_entries(base, lin, group, bits, embed_q[eb], emb_group, eb), r + margin)
            params = dict(model.named_parameters())
            for n, t in restore_emb.items():
                if n in params:
                    params[n].data.copy_(t.to(params[n].dtype).to(params[n].device))
            if deadline.left() < 300:
                break
        if auto:
            nxt = (bits + 1, group) if r_min + margin > target["knee"] else (bits - 1, group)
            if 3 <= nxt[0] <= 8 and nxt not in tried and not (nxt[0] < bits and r_min + margin > target["knee"]):
                # going down only from a candidate under the knee; going up only while over it
                if nxt[0] > bits or best["name"] and best["name"].startswith(f"gptq-w{bits}g"):
                    cands.append(nxt)
    log.info("final: %s, %s bytes, predicted r %s, predicted S %.1f MB, %.0fs elapsed", best["name"], best["bytes"], best["r"], best["S"] / 1e6, deadline.elapsed())
    del model
    gc.collect()


def run_ecq(args, deadline, target, base, names_2d, consider, best, margin, device, src, tgt=None) -> None:
    """ECQ (quant/ecq.py): GPTQ onto per-row uniform grids, steps = c * Fisher allocation, c searched so the
    calibration r lands just under the knee. Each probe is a full candidate (written when its predicted S is best)."""
    t_start = time.time()
    # Hessian accumulation X^T X (fp32, e.g. 18944-wide down_proj inputs over ~0.5M tokens per layer) dominates a GPTQ probe on
    # 7-8B models; TF32 tensor cores run it several times faster (H100: ~7x the fp32 rate). GPTQ is damped, so the reduced mantissa
    # in the Hessian does not matter. Freeform only: the fixed track keeps full fp32.
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("QUANT_ECQ_TF32", "1") == "1"
    tok = common.load_tokenizer(src)
    calib, _ = load_calibration(os.environ.get("QUANT_CALIBRATION_PATH"))
    dataset_type = json.loads(args.dataset_type or "{}")
    max_len = int(os.environ.get("QUANT_MAX_LEN", 2048))
    train = load_train_rows(args.task_id, args.dataset, dataset_type, tok, max_len, int(os.environ.get("QUANT_TRAIN_ROWS", 2000)))
    n_eval = int(os.environ.get("QUANT_EVAL_ROWS", 400))
    eval_rows = [r for r in calib if r.n_scored > 0][:n_eval]
    if not eval_rows:
        eval_rows, train = train[-min(200, len(train) // 4) :], train[: -min(200, len(train) // 4)]
    model = common.load_model(src, device_map=device)
    with torch.no_grad():
        ref_rows = row_losses(model, eval_rows)
    ref_loss = _finite_mean(ref_rows)
    log.info("bf16 loss on %d eval rows: %.5f", len(eval_rows), ref_loss)
    n_test = int(os.environ.get("QUANT_FF_TEST_ROWS", 1000))
    sigma_floor = float(os.environ.get("QUANT_FF_SIGMA_FLOOR", 0.0015))
    bias = float(os.environ.get("QUANT_FF_BIAS", 0.001))  # measured mean of (r_test - r_cal) over 8 local runs: +0.0011 (sd 0.0024)

    def measure() -> tuple[float, float]:
        """(calibration r, sigma of the test r around it) for the weights now in the model."""
        with torch.no_grad():
            rl = row_losses(model, eval_rows)
        d = [a - b for a, b in zip(rl, ref_rows) if a == a and b == b]
        return _finite_mean(rl) / ref_loss, landing.sigma_of(d, ref_loss, n_test, sigma_floor)
    pool = train if len(train) >= 128 else calib + train
    samples = [torch.tensor(r.input_ids) for r in pool[: int(os.environ.get("QUANT_GPTQ_ROWS", 256))]]
    embed_names = [n for n in names_2d if is_embed_like(n)]
    params = dict(model.named_parameters())
    rms = {n: base.tensor(n).float().pow(2).mean(1).sqrt() for n in names_2d}

    alloc_kind = os.environ.get("QUANT_ECQ_ALLOC", "fisher_row")
    log.info("ECQ alloc %s", alloc_kind)
    fisher = None
    tensor_level = None
    if alloc_kind.startswith("fisher"):
        t0 = time.time()
        # lever F (land6): the Fisher that shapes the step allocation was the weakest estimate of the pipeline. (a) rows were cut
        # at 1024 tokens and a row with no scored token before the cut was dropped: t23 (prompts ~1430 tokens, as the R1 task)
        # used 57 of 128 rows, and only their head; (b) 128 rows leave a log-step noise of sd ~0.19 per row (t15: two
        # disjoint 128-row sets differ by sd 0.27), ~0.05 bit per weight by the high-rate rule; (c) the input model was
        # fine-tuned on the train split (t15 bf16 loss 0.90 on train rows vs 1.51 on calibration rows), and the train-row
        # Fisher puts the early blocks 20-30% too coarse and the late blocks 5-20% too fine against held-out rows.
        # Now: QUANT_ECQ_FISHER_ROWS (512) whole rows (QUANT_ECQ_FISHER_LEN 2048), bounded by QUANT_ECQ_FISHER_SECONDS.
        # QUANT_ECQ_FISHER_CALIB=1 (OFF: each tensor's LEVEL (mean Fisher) from calibration rows, row shape from the train
        # rows) lost the RTN equal-r test on the t15 test rows (+0.4..+1.3% bytes vs the train level; 1024 train rows
        # alone: -2.0..+0.2%), so (c) stays unused.
        f_len = int(os.environ.get("QUANT_ECQ_FISHER_LEN", 2048))
        f_secs = float(os.environ.get("QUANT_ECQ_FISHER_SECONDS", 180))
        frows = [r for r in pool if r.n_scored > 0][-int(os.environ.get("QUANT_ECQ_FISHER_ROWS", 512)) :]
        fisher = ecq.fisher_rows(model, frows, names_2d, len(frows), f_len, seconds=f_secs)
        torch.cuda.empty_cache()
        log.info("Fisher in %.0fs", time.time() - t0)
        if os.environ.get("QUANT_ECQ_FISHER_CALIB", "0") == "1" and calib and len(eval_rows) >= 64:
            t0 = time.time()
            crows = eval_rows[: int(os.environ.get("QUANT_ECQ_FISHER_CALIB_ROWS", 256))]
            f_cal = ecq.fisher_rows(model, crows, names_2d, len(crows), f_len, seconds=f_secs / 2)
            mix = float(os.environ.get("QUANT_ECQ_FISHER_CALIB_MIX", 1.0))  # 1 = calibration level, 0 = train level (log scale)
            tensor_level = {}
            for n, f in f_cal.items():
                a, b = float(f.mean()), float(fisher[n].mean()) if n in fisher else 0.0
                if a > 0 and b > 0:
                    tensor_level[n] = math.exp(mix * math.log(a) + (1.0 - mix) * math.log(b))
            sh = sorted(0.5 * math.log(float(fisher[n].mean()) / v) for n, v in tensor_level.items())
            if sh:
                log.info("Fisher level from %d calibration rows in %.0fs: log step shift vs train level min %.2f median %.2f max %.2f (mix %.2f)",
                         len(crows), time.time() - t0, sh[0] - sh[len(sh) // 2], 0.0, sh[-1] - sh[len(sh) // 2], mix)
            del f_cal
            torch.cuda.empty_cache()
    embed_row = os.environ.get("QUANT_ECQ_EMBED_ROW", "1") == "1"
    row_level = set() if alloc_kind != "fisher_row" else {n for n in names_2d if embed_row or n not in embed_names}
    clamp = {n: float(os.environ.get("QUANT_ECQ_EMBED_CLAMP" if n in embed_names else "QUANT_ECQ_ROW_CLAMP", 4.0)) for n in names_2d}
    alloc = ecq.Allocation(base.shapes, rms, fisher, row_level, clamp, tensor_level=tensor_level)
    alloc.set_floor({n: base.tensor(n).float().abs().amax(1) for n in names_2d}, float(os.environ.get("QUANT_ECQ_MAX_CODE", 100)))
    # Vocabulary trim (QUANT_FF_VOCAB_TRIM=1): embedding / head rows of tokens that never occur in the task's data are
    # quantised to zero (step far above the row's range): an input row that is never looked up costs nothing, and a zero
    # head row gives that token logit 0 instead of its trained value. Measured on the calibration rows like everything else.
    # Vocabulary trim (QUANT_FF_VOCAB_TRIM=1): embedding / head rows of tokens that never occur in the task's train split
    # become the mean of those rows (rarely trained tokens sit near it), on one shared step: identical code rows, stored
    # contiguously by the row sort, cost ~nothing under LZ, and their logits stay close to the trained ones. Kept only if
    # its own measured cost at bf16 (all else unquantised) is at most QUANT_FF_TRIM_MAX in r: t15 (small train split, 19k of
    # 152k ids) paid +0.034, t23 / t05 (49k ids) much less. Calibration rows are NOT added to the used set, so they see
    # unseen-token misses at the rate the scored rows will.
    unseen: dict[str, torch.Tensor] = {}
    unseen_base0: dict[str, torch.Tensor] = {}
    used = None
    trimmed: dict[str, torch.Tensor] = {}  # name -> rows replaced by the mean row (trim kept)
    trim_state: dict[str, dict] = {}  # name -> what the kept trim set (override, base, floor) and the untrimmed values
    if os.environ.get("QUANT_FF_VOCAB_TRIM", "1") == "1" and embed_names:
        t0 = time.time()
        used = used_token_ids(args.task_id, args.dataset, tok, [], float(os.environ.get("QUANT_FF_VOCAB_SECONDS", 240)))
        plan = {}
        if used:
            for n in embed_names:
                rows_n = base.shapes[n][0]
                mask = torch.ones(rows_n, dtype=torch.bool)
                ids = torch.tensor(sorted(i for i in used if 0 <= i < rows_n), dtype=torch.long)
                mask[ids] = False
                if mask.any() and not mask.all() and n in params:
                    w = base.tensor(n)
                    w[mask] = w[mask].float().mean(0).to(w.dtype)
                    plan[n] = (mask, w)
        if plan:
            for n, (mask, w) in plan.items():
                params[n].data.copy_(w.to(params[n].dtype).to(params[n].device))
            r_trim, s_trim = measure()
            # keep when the bytes it frees beat what its loss costs at the landing: freed share ~ trimmed embedding weights /
            # all 2-D weights x 0.8 (embeddings code at ~0.8x the mean rate), cost = k x (r_trim - 1), k = QUANT_FF_TRIM_K
            # (d ln bytes / dr near the knee, measured 3.6-11.5). t05: 0.17 vs 0.029 keep (measured S -8%); t23 0.10 vs 0.08
            # keep (measured neutral); t15 0.13 vs 0.23 reject (measured +0.034 r).
            n_all = sum(math.prod(base.shapes[n]) for n in names_2d)
            freed = 0.8 * sum(int(m.sum()) * base.shapes[n][1] for n, (m, _) in plan.items()) / max(1, n_all)
            cost = float(os.environ.get("QUANT_FF_TRIM_K", 8.0)) * max(0.0, r_trim - 1.0)
            keep_trim = freed > cost and r_trim - 1.0 <= float(os.environ.get("QUANT_FF_TRIM_MAX", 0.02))
            log.info("vocab trim: frees ~%.3f of the bytes, costs ~%.3f at the landing", freed, cost)
            log.info("vocab trim: %d token ids used, %s rows trimmed; bf16 + trim r %.4f (sigma %.4f): %s (%.0fs)", len(used),
                     ",".join(str(int(m.sum())) for m, _ in plan.values()), r_trim, s_trim, "kept" if keep_trim else "REJECTED", time.time() - t0)
            if keep_trim:
                base.overrides = {n: w for n, (mask, w) in plan.items()}
                for n, (mask, w) in plan.items():
                    shared = alloc.base[n][~mask].median()
                    trim_state[n] = {"base0": alloc.base[n].clone(), "floor0": alloc.floor[n].clone() if n in alloc.floor else None}
                    alloc.base[n] = torch.where(mask, shared.expand_as(alloc.base[n]), alloc.base[n])
                    if n in alloc.floor:
                        alloc.floor[n] = torch.where(mask, alloc.floor[n][mask].max().expand_as(alloc.floor[n]), alloc.floor[n])
                    trimmed[n] = mask
                    trim_state[n].update(over=w, base=alloc.base[n].clone(), floor=alloc.floor[n].clone() if n in alloc.floor else None)
            else:
                for n in plan:
                    params[n].data.copy_(base.tensor(n).to(params[n].dtype).to(params[n].device))
                # lever E1 (QUANT_FF_EMB_UNSEEN=1): the trim is rejected (t15: 133k of 152k rows never occur in the train split,
                # yet 0.7% of the scored tokens are such ids), and the 128 Fisher rows see so few tokens that 98.8% of the
                # embedding rows sit at the coarse row clamp on one shared step (F_t15_a: 150k rows, 70 MB, 2.43 bits per
                # weight; the unseen rows alone ~62 MB for 0.7% of the tokens). Keep the unseen mask: at GPTQ probe 0 the
                # unseen rows' step is scanned x U (one eval each) and the U with the best block-equivalent score is kept.
                if os.environ.get("QUANT_FF_EMB_UNSEEN", "1") == "1":
                    unseen = {n: m for n, (m, _) in plan.items()}
                    unseen_base0 = {n: alloc.base[n].clone() for n in unseen}

    # lever E2 (land6, QUANT_FF_EMB_GPTQ=1): rows of the output embedding (the tied table, or an untied lm_head) can be
    # quantised by GPTQ against the head's input second moment H = E[h h^T] (final hidden states at scored positions)
    # instead of RTN. H is very anisotropic (26 Nov: effective rank 4.5 on t15, 7.5 on t23, 23 on t05), so error feedback
    # cuts a row's logit error variance d^T H d 4-7x at the same step. Measured on the shipped 864ff23 artifacts, r on the
    # calibration rows: it pays where rows can then take a much coarser step:
    #   - rare rows (train-split count 1..QUANT_FF_EMB_RARE_MAX): step x3 on t23 -7.3 MB for +0.0004 r (RTN +0.0022);
    #   - "soft trim" when the vocab trim is kept: the unseen rows from their ORIGINAL values at a very coarse step instead
    #     of the mean row: t23 at 2.9 x the rows' rms +5.5 MB for -0.0064 r (RTN at 2.2 x rms: +7.2 MB for -0.0026), t05
    #     no option beats the trim: the options are scanned at GPTQ probe 0 and the trim stays one of them;
    #   - the unseen rows when the trim is rejected (lever E1's rows): x1 -0.0005 r, x2 half the RTN cost (t15).
    # Rows with more occurrences stay RTN: they are read on the input side at every occurrence, where the error is |d|^2.
    emb_gptq: dict[str, torch.Tensor] = {}  # name -> bool [rows]: rows quantised by GPTQ against head_H
    rare: dict[str, torch.Tensor] = {}
    rare_base0: dict[str, torch.Tensor] = {}
    head_H = None
    emb_damp = float(os.environ.get("QUANT_FF_EMB_GPTQ_DAMP", 0.1))
    if os.environ.get("QUANT_FF_EMB_GPTQ", "1") == "1" and embed_names and used and model.get_output_embeddings() is not None:
        t0 = time.time()
        hw = model.get_output_embeddings().weight
        head_name = next((n for n in embed_names if n in params and params[n] is hw), None)
        if head_name is not None:
            head_H = head_hessian(model, [r for r in pool if r.n_scored > 0], int(os.environ.get("QUANT_GPTQ_ROWS", 256)))
        if head_H is not None:
            rows_n = base.shapes[head_name][0]
            cnt = torch.zeros(rows_n, dtype=torch.long)
            for i, v in used.items():
                if 0 <= i < rows_n:
                    cnt[i] = min(int(v), SPECIAL_COUNT)
            rare[head_name] = (cnt >= 1) & (cnt <= int(os.environ.get("QUANT_FF_EMB_RARE_MAX", 16)))
            rare_base0[head_name] = alloc.base[head_name].clone()
            emb_gptq[head_name] = rare[head_name].clone()
            if head_name in unseen:
                emb_gptq[head_name] |= unseen[head_name]
            ev = torch.linalg.eigvalsh(head_H.double())
            log.info("EMBGPTQ: head %s, H from the scored positions of the GPTQ rows (effective rank %.1f), %d rare rows (count 1..%s), %d rows by GPTQ, "
                     "damp %.2g (%.0fs)", head_name, float(ev.sum() ** 2 / ev.pow(2).sum()), int(rare[head_name].sum()),
                     os.environ.get("QUANT_FF_EMB_RARE_MAX", "16"), int(emb_gptq[head_name].sum()), emb_damp, time.time() - t0)
    emb_cache: dict = {}

    def q_emb(n: str, st: torch.Tensor):
        """Quantiser of an embedding-like tensor at per-row steps st: RTN, and the rows in emb_gptq[n] by GPTQ against
        head_H (the last result is cached: the embeddings keep their steps over the GPTQ probes)."""
        w = base.tensor(n)
        res = ecq.rtn_u(w, st, device=device)
        m = emb_gptq.get(n)
        if head_H is None or m is None or not bool(m.any()):
            return res
        key = (n, int(m.sum()), hash(st.float().numpy().tobytes()), hash(m.numpy().tobytes()), id(getattr(base, "overrides", {}).get(n)))
        if emb_cache.get("key") == key:
            return emb_cache["res"]
        g = ecq.gptq_u(w[m], head_H, st[m], damp=emb_damp, act_order=True)
        codes = res.codes.clone()
        codes[m] = g.codes
        out = ecq.UResult(codes, res.step)
        emb_cache.update(key=key, res=out)
        return out

    aim = target["knee"] - margin - float(os.environ.get("QUANT_ECQ_UNDERSHOOT", 0.001))
    size_pts: list[tuple[float, int]] = []  # (r, bytes) of encoded candidates: slope of ln bytes in r
    sigma_now = float(os.environ.get("QUANT_FF_SIGMA0", 0.0025))
    tune_f = float(os.environ.get("QUANT_FF_TUNE_F", 0.0)) if os.environ.get("QUANT_FF_TUNE", "0") == "1" else 0.0
    pre = (lambda a: 1.0 + (a - 1.0) / (1.0 - tune_f)) if tune_f > 0 else (lambda a: a)
    if tune_f > 0:  # rows for tuning: train rows minus a held-out selection slice
        n_sel = min(128, max(16, len(train) // 10))
        tune_sel, tune_train = train[-n_sel:], train[:-n_sel]
    if tgt is not None:
        aim = pre(tgt.aim(sigma_now, float(os.environ.get("QUANT_FF_SLOPE0", -8.0))) - bias)
        log.info("landing aim r_cal %.4f (sigma %.4f, prior slope)", aim, sigma_now)
    # start: geometric-mean step / weight RMS = 0.2 (~4.4 bits/weight for Gaussian weights); the empirical Fisher of a
    # row's mean loss is ~1/tokens of the per-token curvature, so its absolute scale is not used, only its shape
    c = alloc.c_for_ratio(float(os.environ.get("QUANT_ECQ_STEP_OVER_RMS", 0.2)), rms)
    if os.environ.get("QUANT_ECQ_C0"):
        c = float(os.environ["QUANT_ECQ_C0"])

    def restore():
        for n in names_2d:
            if n in params:
                params[n].data.copy_(base.tensor(n).to(params[n].dtype).to(params[n].device))

    damp = float(os.environ.get("QUANT_DAMP", 0.01))
    col_alpha = float(os.environ.get("QUANT_ECQ_COL_ALPHA", 0.0))  # >0: per-column step factors from the Hessian diagonal
    col_clamp = float(os.environ.get("QUANT_ECQ_COL_CLAMP", 4.0))
    max_code = float(os.environ.get("QUANT_ECQ_MAX_CODE", 100))
    rdo = float(os.environ.get("QUANT_ECQ_RDO", 0.0))
    # L5 dependent quantisation (quant/dq.py, QUANT_ECQ_DQ=1): decoder Linears get a 4-state trellis on the step
    # D = g x (the scalar step), floored at |w|max / max_code so codes stay inside int8; embeddings stay scalar.
    # on by default since the paired t23 GPU runs passed (25 Nov: fixed c, DQ 4.4% smaller at 0.005 lower r); QUANT_ECQ_DQ=0
    # gives the ed1b9a6 scalar path
    use_dq = os.environ.get("QUANT_ECQ_DQ", "1") == "1"
    dq_g = float(os.environ.get("QUANT_ECQ_DQ_G", 0.6))
    dq_lam = float(os.environ.get("QUANT_ECQ_DQ_LAM", 0.15))  # rate term of the trellis path metric (microtest: -0.5..-1.5% more than 0)
    dq_block = int(os.environ.get("QUANT_ECQ_DQ_BLOCK", 128))
    dq_rate_cols = int(os.environ.get("QUANT_ECQ_DQ_RATE_COLS", 512))  # rate tables from the first columns only (time)
    # trellis column order: "b16" = act order in 16 buckets (4 bits per column in the artifact; microtest: better than
    # the exact act order once its permutation bytes count), "act" = exact act order, "nat" = natural order
    dq_order = {"b16": "b16", "act": True, "nat": False}[os.environ.get("QUANT_ECQ_DQ_ORDER", "b16")]
    if os.environ.get("QUANT_ECQ_ACT_ORDER", "1") != "1":
        dq_order = False
    # time kept free for the final state-aware coding of trellis codes (final_rans): codes / rate + verify
    dq_reserve = (sum(math.prod(base.shapes[n]) for n in names_2d) / float(os.environ.get("QUANT_FF_DQ_RATE", 8e6)) + 60) if use_dq else 0.0
    if use_dq:
        log.info("ECQ: dependent quantisation on (g %.3f, lambda %.3f, block %d), final coding reserve %.0fs", dq_g, dq_lam, dq_block, dq_reserve)

    def dq_step(w, st, col=None):
        d = st.float().cpu() * dq_g
        a = w.float().abs()
        if col is not None:
            a = a / col.to(a.device).unsqueeze(0)
        return torch.maximum(d, (a.amax(1) / max_code).cpu())

    def rtn_lin(n, w, st):
        """RTN-level quantiser of a 2-D weight. Scalar by default even with DQ on: at D = g x step (g 0.6) the GPTQ
        trellis lands at about the scalar GPTQ error (microtest), so the scalar RTN bracket still places the GPTQ probes,
        at none of the trellis cost. QUANT_ECQ_DQ_RTN=1: trellis without error feedback."""
        if use_dq and os.environ.get("QUANT_ECQ_DQ_RTN", "0") == "1" and not is_embed_like(n):
            return dq.rtn_dq(w, dq_step(w, st), device=device)
        return ecq.rtn_u(w, st, device=device)
    history: list[tuple[float, float]] = []
    # Landing fix (QUANT_FF_EMB_OWN_C=1, default): the embeddings / head get their own step constant, fixed at the first
    # GPTQ probe's c (times QUANT_FF_EMB_C_FACTOR); later probes move the decoder blocks alone. On t23 (2bfa93f, 24 Nov)
    # the scalar tied embedding alone moved r +0.0126 between two probes 1.25x apart in c while the 28 blocks moved it
    # +0.001, so the search bracketed early on the embedding's cliff and stopped ~0.010 r under the aim with ~35 min unused.
    # With the embedding held, the GPTQ history is the blocks' own r(c) curve (every probe shares the embedding).
    emb_own_c = os.environ.get("QUANT_FF_EMB_OWN_C", "1") == "1" and bool(embed_names)
    emb_c_factor = float(os.environ.get("QUANT_FF_EMB_C_FACTOR", 1.0))
    emb_scan = emb_own_c and os.environ.get("QUANT_FF_EMB_SCAN", "1") == "1"
    max_probes = int(os.environ.get("QUANT_ECQ_PROBES", 8 if emb_own_c else 5))
    tol = float(os.environ.get("QUANT_ECQ_TOL", 0.0015))
    probe_time = None
    log.info("ECQ: alloc %s, aim r %.4f, c0 %.4g, setup %.0fs", alloc.kind, aim, c, time.time() - t_start)

    def secant(hist, target, c_now, r_now, jump):
        a = 2.0
        pts = [(math.log(cc), math.log(max(rr - 1.0, 1e-5))) for cc, rr in hist]
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            if abs(x2 - x1) > 1e-6:
                a = min(4.0, max(1.0, (y2 - y1) / (x2 - x1)))
        c_new = c_now * math.exp((math.log(max(target - 1.0, 1e-5)) - math.log(max(r_now - 1.0, 1e-5))) / a)
        return min(c_now * jump, max(c_now / jump, c_new))

    def rtn_curve(hist):
        """least-squares line log(r - 1) = a log c + b over the RTN probes (None with < 2 usable points)."""
        pts = [(math.log(cc), math.log(rr - 1.0)) for cc, rr in hist if rr - 1.0 > 0.002]
        near = [(x, y) for x, y in pts if 0.008 < math.exp(y) < 0.1]  # the curve bends: fit the part near the aim
        if len(near) >= 2:
            pts = near
        if len(pts) < 2:
            return None
        mx = sum(x for x, _ in pts) / len(pts)
        my = sum(y for _, y in pts) / len(pts)
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        if sxx < 1e-9:
            return None
        a = sum((x - mx) * (y - my) for x, y in pts) / sxx
        return max(1.0, min(6.0, a)), my - max(1.0, min(6.0, a)) * mx

    def next_gptq_c(hist, rhist, target, c_now, r_now):
        """GPTQ's r(c) is the RTN curve shifted to larger c: estimate the shift from the latest GPTQ probe, use the RTN
        slope (models differ a lot: ~2 on 0.5-1.5B fine-tunes, ~4.6 on Qwen3-8B-Base), and move at most 1.25x."""
        # with the embeddings held (emb_own_c) the RTN curve, which moved them too, is not the blocks' curve: secant on the
        # GPTQ probes alone (slope prior 2), at most 1.25x per probe
        line = None if emb_own_c else rtn_curve(rhist)
        if line is None:
            return secant(hist, target, c_now, r_now, 1.25 if emb_own_c else 1.5)
        a, _b = line
        if len(hist) >= 2:  # GPTQ's own slope once two GPTQ points exist, bounded by the RTN slope's range
            (x1, y1), (x2, y2) = [(math.log(cc), math.log(max(rr - 1.0, 1e-5))) for cc, rr in hist[-2:]]
            if abs(x2 - x1) > 1e-6:
                a = max(1.0, min(6.0, (y2 - y1) / (x2 - x1)))
        c_new = c_now * math.exp((math.log(max(target - 1.0, 1e-5)) - math.log(max(r_now - 1.0, 1e-5))) / a)
        return min(c_now * 1.25, max(c_now / 1.25, c_new))

    def group_key(n: str) -> str:
        leaf = n.rsplit(".", 2)[-2] if n.count(".") >= 1 else n
        return ("embed:" + leaf) if is_embed_like(n) else leaf

    def measured_group_allocation(c_m: float, t_rtn: float | None) -> bool:
        """Measured re-allocation between tensor groups (q/k/v/o/gate/up/down/embeddings, by module name). The Fisher
        allocation assumes loss ~ f * step^2 everywhere; on fine-tuned models the measured marginal cost differs by
        5-10x between groups (t15: k_proj 0.33, o_proj 2.5 MB per 0.001 r; the tied embedding ~free). At c_m, each group's
        steps are scaled by F alone (RTN onto the grid, one eval each), giving its bytes saved per unit r; groups then get
        factor (e_g / e_mean)^alpha (clamped), which moves the allocation toward equal marginal cost. Returns True if
        the allocation changed."""
        groups: dict[str, list[str]] = {}
        for n in names_2d:
            groups.setdefault(group_key(n), []).append(n)
        if len(groups) < 2:
            return False
        need = (t_rtn or 60.0) * (len(groups) + 1) * 1.5 + 60
        if deadline.left() < need + float(os.environ.get("QUANT_FF_MGA_RESERVE", 1500)):
            log.info("MGA: skipped (%.0fs left, needs ~%.0fs)", deadline.left(), need)
            return False
        t0 = time.time()
        F = float(os.environ.get("QUANT_FF_MGA_FACTOR", 1.4))
        alpha = float(os.environ.get("QUANT_FF_MGA_ALPHA", 0.5))
        lo_f, hi_f = float(os.environ.get("QUANT_FF_MGA_MIN", 0.5)), float(os.environ.get("QUANT_FF_MGA_MAX", 2.0))
        steps0 = alloc.steps(c_m)
        from concurrent.futures import ThreadPoolExecutor

        def sizes(res: dict) -> dict[str, int]:
            with ThreadPoolExecutor(16) as ex:
                return dict(zip(res, ex.map(lambda q: len(ffcodec._compress(q.codes.numpy().tobytes(), 0, True)), res.values())))

        restore()
        res0 = {n: ecq.rtn_u(params[n].data, steps0[n], device=device) for n in names_2d if n in params}
        set_weights(model, res0, 0)
        with torch.no_grad():
            r0 = mean_loss(model, eval_rows) / ref_loss
        size0 = sizes(res0)
        del res0
        eff: dict[str, float] = {}
        gbytes: dict[str, int] = {}
        for g, ns in groups.items():
            ns = [n for n in ns if n in params]
            if not ns:
                continue
            restore()
            res = {n: ecq.rtn_u(params[n].data, steps0[n] * (F if n in ns else 1.0), device=device) for n in names_2d if n in params}
            set_weights(model, res, 0)
            with torch.no_grad():
                r1 = mean_loss(model, eval_rows) / ref_loss
            s1 = sizes({n: res[n] for n in ns})
            del res
            dB = sum(size0[n] - s1[n] for n in ns)
            dr = r1 - r0
            gbytes[g] = sum(size0[n] for n in ns)
            eff[g] = dB / max(dr, 2e-4)  # bytes saved per unit r (a change under the noise counts as 0.0002)
            log.info("MGA %s: %.1f MB, x%.2f steps: -%.2f MB, dr %+.4f -> %.2f MB per 0.001 r", g, gbytes[g] / 1e6, F, dB / 1e6, dr, eff[g] / 1e9)
        pos = {g: e for g, e in eff.items() if e > 0}
        if len(pos) < 2:
            return False
        tot = sum(gbytes[g] for g in pos)
        gm = math.exp(sum(gbytes[g] * math.log(e) for g, e in pos.items()) / tot)
        changed = False
        for g, e in pos.items():
            f = min(hi_f, max(lo_f, (e / gm) ** alpha))
            if abs(f - 1.0) > 0.02:
                changed = True
                for n in groups[g]:
                    alloc.base[n] = alloc.base[n] * f
            log.info("MGA factor %s: %.3f", g, f)
        log.info("MGA: base r %.4f, %d groups, %.0fs", r0, len(eff), time.time() - t0)
        return changed

    # Cheap bracketing with RTN onto the same grids (seconds per probe instead of minutes): GPTQ then starts at the
    # RTN landing point, where its r is lower (safe side), and the secant steps up from there.
    n_rtn = int(os.environ.get("QUANT_ECQ_RTN_PROBES", 4))
    rtn_hist: list[tuple[float, float]] = []
    mga_done = False
    rtn_time = None
    k = 0
    while k < n_rtn:
        if deadline.left() < 600:
            break
        t0 = time.time()
        restore()
        steps = alloc.steps(c)
        res = {n: rtn_lin(n, params[n].data, steps[n]) for n in names_2d if n in params}
        set_weights(model, res, 0)
        del res
        with torch.no_grad():
            r = mean_loss(model, eval_rows) / ref_loss
        rtn_hist.append((c, r))
        rtn_time = time.time() - t0
        log.info("ECQ RTN probe %d: c %.5g r %.4f (%.0fs)", k, c, r, rtn_time)
        k += 1
        if abs(r - aim) <= 0.003 or k >= n_rtn:
            if not mga_done and tgt is not None and os.environ.get("QUANT_FF_MGA", "0") == "1":
                mga_done = True
                c_m = min(rtn_hist, key=lambda x: abs(x[1] - aim))[0]
                if measured_group_allocation(c_m, rtn_time):
                    # the allocation changed: bracket again from the same c
                    rtn_hist.clear()
                    k = 0
                    c = c_m
                    continue
            break
        line = rtn_curve(rtn_hist)
        if line is None:
            c = secant(rtn_hist, aim, c, r, 8.0)
        else:
            a, b = line
            c = min(c * 8.0, max(c / 8.0, math.exp((math.log(max(aim - 1.0, 1e-5)) - b) / a)))
    if rtn_hist:
        # the RTN point closest to the aim from below, else the smallest c tried
        under = [(cc, rr) for cc, rr in rtn_hist if rr <= aim + 0.003]
        c = max(under)[0] if under else min(rtn_hist)[0]
        line = rtn_curve(rtn_hist)
        if line is not None:  # the fitted RTN line at the aim is less noisy than any single probe
            a, b = line
            # first GPTQ probe aims below the aim (more so on large models, where only ~2 GPTQ probes fit a short budget):
            # if the second probe overshoots, the first is still a valid under-knee result
            first_aim = aim - float(os.environ.get("QUANT_ECQ_FIRST_UNDER", 0.005 if base.bf16_bytes() > 8e9 else 0.002))
            c_fit = math.exp((math.log(max(first_aim - 1.0, 1e-5)) - b) / a)
            c = min(c * 1.3, max(c / 1.3, c_fit))
        c *= float(os.environ.get("QUANT_ECQ_GPTQ_GAIN", 1.05))
    final = None  # landing mode: the shipped candidate's codes (for the optional tuning stage)
    keep: dict[str, dict] = {}  # landing mode: the closest probe under the aim ("lo") and over it ("hi"), codes kept for mixing

    def bracket_next(c_now: float) -> float | None:
        lo, hi = keep.get("lo"), keep.get("hi")
        if lo is None or hi is None:
            return None
        x1, y1 = math.log(lo["c"]), math.log(max(lo["r"] - 1.0, 1e-5))
        x2, y2 = math.log(hi["c"]), math.log(max(hi["r"] - 1.0, 1e-5))
        if abs(y2 - y1) < 1e-9:
            return None
        x = x1 + (math.log(max(aim - 1.0, 1e-5)) - y1) * (x2 - x1) / (y2 - y1)
        x = min(x1 + 0.8 * (x2 - x1), max(x1 + 0.2 * (x2 - x1), x))  # stay inside the bracket
        return math.exp(x)

    def name_bytes(p: dict, n: str) -> int:
        q = p["lin"].get(n)
        if q is not None and getattr(q, "dq", False) and isinstance(getattr(q, "est", None), int):
            return q.est
        return sum(len(f) for f in p["enc"][n][1])

    def unit_swaps(lo: dict, hi: dict) -> dict:
        """Instrumentation for the measured per-unit allocation (lever 2, QUANT_FF_UNIT_SWAP=1): from the lo probe, swap
        ONE unit at a time to the hi probe's codes and measure r on the calibration rows (paired rows), giving each
        unit's bytes saved and r cost. Units: the decoder blocks, then the module groups (q/k/v/o/gate/up/down across
        all blocks), then all at once (additivity check). One "UNITSWAP {json}" line per eval; the model is left at lo."""
        names = [n for n in names_2d if n in params]
        sets = []
        for b in sorted({unit_of(n) for n in names if unit_of(n) >= 0}):
            sets.append((f"block{b}", [n for n in names if unit_of(n) == b]))
        only_blocks = os.environ.get("QUANT_FF_UNIT_SWAP_SETS", "all") == "blocks"
        for g in ([] if only_blocks else sorted({group_key(n) for n in names if unit_of(n) >= 0})):
            sets.append((f"group:{g}", [n for n in names if unit_of(n) >= 0 and group_key(n) == g]))
        emb = [n for n in names if unit_of(n) < 0]
        if emb and any(not torch.equal(lo["lin"][n].codes, hi["lin"][n].codes) for n in emb if n in lo["lin"] and n in hi["lin"]):
            sets.append(("emb", emb))
        if not only_blocks:
            sets.append(("all", names))
        meas = {}
        lo_b = {n: name_bytes(lo, n) for n in names}
        hi_b = {n: name_bytes(hi, n) for n in names}
        set_weights(model, {n: lo["lin"][n] for n in names}, 0)
        r0, s0 = measure()
        log.info("UNITSWAP %s", json.dumps({"unit": "lo", "r": r0, "c_lo": lo["c"], "c_hi": hi["c"], "r_lo_probe": lo["r"], "r_hi_probe": hi["r"],
                                            "bytes_lo": sum(lo_b.values()), "bytes_hi": sum(hi_b.values())}))
        t_sw = time.time()
        for u, ns in sets:
            if deadline.left() < 600 + dq_reserve:
                log.info("UNITSWAP: out of time at %s", u)
                break
            t0 = time.time()
            set_weights(model, {n: hi["lin"][n] for n in ns}, 0)
            r1, _ = measure()
            set_weights(model, {n: lo["lin"][n] for n in ns}, 0)
            meas[u] = (r1 - r0, sum(lo_b[n] - hi_b[n] for n in ns))
            log.info("UNITSWAP %s", json.dumps({"unit": u, "n": len(ns), "r": r1, "dr": r1 - r0, "dB": sum(lo_b[n] - hi_b[n] for n in ns),
                                                "B_lo": sum(lo_b[n] for n in ns), "s": round(time.time() - t0, 1)}))
        log.info("UNITSWAP done: %d sets in %.0fs", len(sets), time.time() - t_sw)
        return meas

    # lever 2b (QUANT_FF_BLOCK_REALLOC=1): once a lo / hi bracket exists, measure each decoder block's swap lo -> hi alone
    # (bytes saved, r cost on the paired calibration rows) and re-allocate: block b's steps x (eff_b / mean eff)^alpha,
    # clamped, normalised so the bytes-weighted log factor is 0 (bytes ~unchanged); the search then restarts from the
    # bracket's middle c on the new allocation. t15 (25 Nov): single-block costs ranged -0.0006..+0.0020 r for ~0.63 MB each.
    realloc = os.environ.get("QUANT_FF_BLOCK_REALLOC", "0") == "1"
    realloc_done = False
    eval_t = None
    picked_over = False

    def split_r() -> tuple[float, float, float]:
        """(r all rows, r even rows, r odd rows) of the weights in the model."""
        with torch.no_grad():
            rl = row_losses(model, eval_rows)
        def rr(ix):
            a = [rl[i] for i in ix if rl[i] == rl[i] and ref_rows[i] == ref_rows[i]]
            b = [ref_rows[i] for i in ix if rl[i] == rl[i] and ref_rows[i] == ref_rows[i]]
            return sum(a) / max(1e-12, sum(b))
        n = len(rl)
        return rr(range(n)), rr(range(0, n, 2)), rr(range(1, n, 2))

    def even_r() -> float:
        """r on the even calibration rows only (half the eval time)."""
        ev = eval_rows[0::2]
        with torch.no_grad():
            rl = row_losses(model, ev)
        ref = ref_rows[0::2]
        a = [x for x, y in zip(rl, ref) if x == x and y == y]
        b = [y for x, y in zip(rl, ref) if x == x and y == y]
        return sum(a) / max(1e-12, sum(b))

    def block_pick(fin: dict, full: bool = False):
        nonlocal eval_t
        t_all = time.time()
        st = fin["steps"]
        off = int(os.environ.get("QUANT_GPTQ_ROWS", 256))
        alt = pool[off : 2 * off] if len(pool) >= off + 64 else pool[::-1][:off]
        samples_b = [torch.tensor(r_.input_ids) for r_ in alt]
        restore()

        def qz(name, w, H):
            if H is None:
                return rtn_lin(name, w, st[name])
            act = os.environ.get("QUANT_ECQ_ACT_ORDER", "1") == "1"
            if use_dq and not is_embed_like(name):
                d = dq_step(w, st[name])
                kw = {}
                if dq_lam > 0:
                    bits, cls = dq.rate_model(w, d, H.device, dq_rate_cols)
                    kw = {"lam": dq_lam, "bits": bits, "cls": cls}
                return dq.gptq_dq(w, H, d, damp=damp, blocksize=dq_block, act_order=dq_order if act else False, **kw)
            return ecq.gptq_u(w, H, st[name], damp=damp, act_order=act, rdo=rdo)

        def gqz(names, ws, H):
            if use_dq and not any(is_embed_like(n) for n in names):
                sizes = [w.shape[0] for w in ws]
                wcat = torch.cat(ws, 0)
                d = torch.cat([dq_step(w, st[n]) for n, w in zip(names, ws)])
                kw = {}
                if dq_lam > 0:
                    bits, cls = dq.rate_model(wcat, d, H.device, dq_rate_cols)
                    kw = {"lam": dq_lam, "bits": bits, "cls": cls}
                u = dq.gptq_dq(wcat, H, d, damp=damp, blocksize=dq_block, act_order=dq_order, split=sizes, **kw)
                out, r0 = [], 0
                for i, sz in enumerate(sizes):
                    out.append(ecq.UResult(u.codes[r0 : r0 + sz].contiguous(), u.step[r0 : r0 + sz].contiguous(), dq=True, perm=u.perm, okey=u.okey,
                                           chunk=u.chunk, est=u.est[i]))
                    r0 += sz
                return out
            return [qz(n, w, H) for n, w in zip(names, ws)]

        lin_b = quantize_model_gptq(model, samples_b, QSpec(bits=8, group=1), damp=damp, true_sequential=True, time_left=lambda: deadline.left() - 300,
                                    quantizer=qz, group_quantizer=gqz if use_dq else None)
        lin_b = {n: q for n, q in lin_b.items() if n in base.shapes and not is_embed_like(n)}
        lin_a = fin["lin"]
        for n in names_2d:
            if n not in lin_b:
                lin_b[n] = lin_a[n]
        names = [n for n in names_2d if n in params]
        set_weights(model, {n: lin_b[n] for n in names}, 0)
        rb_all, rb_e, rb_o = split_r()
        set_weights(model, {n: lin_a[n] for n in names}, 0)
        t0 = time.time()
        ra_all, ra_e, ra_o = split_r()
        eval_t = time.time() - t0
        log.info("BLOCKPICK draw B (%d alt rows): r %.4f (even %.4f odd %.4f) vs A %s r %.4f (even %.4f odd %.4f), B probe+evals %.0fs",
                 len(samples_b), rb_all, rb_e, rb_o, fin["name"], ra_all, ra_e, ra_o, time.time() - t_all)
        pick = []
        for b in sorted({unit_of(n) for n in names if unit_of(n) >= 0}):
            if deadline.left() < 400 + dq_reserve:
                log.info("BLOCKPICK: out of time at block %d", b)
                return None
            ns = [n for n in names if unit_of(n) == b]
            set_weights(model, {n: lin_b[n] for n in ns}, 0)
            re_ = even_r()
            set_weights(model, {n: lin_a[n] for n in ns}, 0)
            if re_ < ra_e:
                pick.append(b)
            log.info("BLOCKPICK block %d: even-row dr %+.5f", b, re_ - ra_e)
        if not pick:
            return None
        comb = {n: (lin_b[n] if unit_of(n) in pick else lin_a[n]) for n in names_2d}
        set_weights(model, {n: comb[n] for n in names}, 0)
        rc_all, rc_e, rc_o = split_r()
        log.info("BLOCKPICK combined (%d blocks from B): r %.4f even %.4f odd %.4f (A odd %.4f): %s", len(pick), rc_all, rc_e, rc_o, ra_o,
                 "accepted" if rc_o < ra_o else "rejected")
        if rc_o >= ra_o:
            return None
        r_, sigma_ = measure()
        entries = [ffcodec.Entry(name, shape, q=comb[name]) if name in comb else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                   for name, shape in base.shapes.items()]
        # the blocks were chosen on the even rows: half of the even-over-odd excess gain is selection optimism in the all-rows r
        corr = max(0.0, ((ra_e - rc_e) - (ra_o - rc_o)) / 2.0) if full else 0.0
        records, frames, est = consider(fin["name"] + f"-pick{len(pick)}", entries, r_ + corr + bias, sigma=sigma_)
        if full:
            return {"lin": comb, "r": r_, "sigma": sigma_, "corr": corr, "pick": pick, "enc": split_encoded(records, frames), "est": est,
                    "name": fin["name"] + f"-pick{len(pick)}", "gain_odd": ra_o - rc_o, "gain_even": ra_e - rc_e}
        return comb, r_

    def gptq_at(st: dict, smp: list) -> dict:
        """One GPTQ(+DQ) pass at the given per-tensor scalar steps (same quantisers as a probe); embeddings stay out."""
        restore()

        def qz(name, w, H):
            if H is None:
                return rtn_lin(name, w, st[name])
            act = os.environ.get("QUANT_ECQ_ACT_ORDER", "1") == "1"
            if use_dq and not is_embed_like(name):
                d = dq_step(w, st[name])
                kw = {}
                if dq_lam > 0:
                    bits, cls = dq.rate_model(w, d, H.device, dq_rate_cols)
                    kw = {"lam": dq_lam, "bits": bits, "cls": cls}
                return dq.gptq_dq(w, H, d, damp=damp, blocksize=dq_block, act_order=dq_order if act else False, **kw)
            return ecq.gptq_u(w, H, st[name], damp=damp, act_order=act, rdo=rdo)

        def gqz(names, ws, H):
            if use_dq and not any(is_embed_like(n) for n in names):
                sizes = [w.shape[0] for w in ws]
                wcat = torch.cat(ws, 0)
                d = torch.cat([dq_step(w, st[n]) for n, w in zip(names, ws)])
                kw = {}
                if dq_lam > 0:
                    bits, cls = dq.rate_model(wcat, d, H.device, dq_rate_cols)
                    kw = {"lam": dq_lam, "bits": bits, "cls": cls}
                u = dq.gptq_dq(wcat, H, d, damp=damp, blocksize=dq_block, act_order=dq_order, split=sizes, **kw)
                out, r0 = [], 0
                for i, sz in enumerate(sizes):
                    out.append(ecq.UResult(u.codes[r0 : r0 + sz].contiguous(), u.step[r0 : r0 + sz].contiguous(), dq=True, perm=u.perm, okey=u.okey,
                                           chunk=u.chunk, est=u.est[i]))
                    r0 += sz
                return out
            return [qz(n, w, H) for n, w in zip(names, ws)]

        out = quantize_model_gptq(model, smp, QSpec(bits=8, group=1), damp=damp, true_sequential=True, time_left=lambda: deadline.left() - 300,
                                  quantizer=qz, group_quantizer=gqz if use_dq else None)
        return {n: q for n, q in out.items() if n in base.shapes and not is_embed_like(n)}

    def block_realloc(lo: dict, hi: dict) -> bool:
        os.environ["QUANT_FF_UNIT_SWAP_SETS"] = "blocks"
        meas = unit_swaps(lo, hi)
        bl = sorted({unit_of(n) for n in names_2d if unit_of(n) >= 0})
        if not all(f"block{b}" in meas for b in bl):
            return False
        floor = float(os.environ.get("QUANT_FF_REALLOC_DR_FLOOR", 2e-4))
        al = float(os.environ.get("QUANT_FF_REALLOC_ALPHA", 0.5))
        lo_f, hi_f = float(os.environ.get("QUANT_FF_REALLOC_MIN", 0.8)), float(os.environ.get("QUANT_FF_REALLOC_MAX", 1.25))
        eff = {b: max(1.0, meas[f"block{b}"][1]) / max(floor, meas[f"block{b}"][0]) for b in bl}
        wts = {b: sum(name_bytes(lo, n) for n in names_2d if unit_of(n) == b) for b in bl}
        tot = sum(wts.values())
        gm = math.exp(sum(wts[b] * math.log(eff[b]) for b in bl) / tot)
        f = {b: min(hi_f, max(lo_f, (eff[b] / gm) ** al)) for b in bl}
        # step factor f -> ~ -log2(f) bits per weight: centre the bytes-weighted log so the total rate stays put
        mlog = sum(wts[b] * math.log(f[b]) for b in bl) / tot
        f = {b: math.exp(math.log(v) - mlog) for b, v in f.items()}
        for n in names_2d:
            b = unit_of(n)
            if b >= 0:
                alloc.base[n] = alloc.base[n] * f[b]
        log.info("block realloc factors: %s", ",".join(f"{b}:{f[b]:.3f}" for b in bl))
        return True

    for k in range(max_probes):
        # GPTQ itself switches to RTN when less than ~1.2 x its time is left beyond a 300 s margin: never start a probe
        # that would degrade that way (it costs time and lands nowhere useful)
        if probe_time is not None and deadline.left() < probe_time * float(os.environ.get("QUANT_ECQ_GATE_MULT", 1.15)) + 330 + dq_reserve:
            log.info("ECQ: no time for another probe (%.0fs left, probe %.0fs)", deadline.left(), probe_time)
            break
        t0 = time.time()
        restore()
        if emb_own_c and k == 0:
            alloc.fixed_c = {n: c * emb_c_factor for n in embed_names}
            log.info("ECQ: embeddings held at c %.5g (%s) for every GPTQ probe", c * emb_c_factor, ",".join(embed_names))
        steps = alloc.steps(c)

        def quantizer(name, w, H):
            if H is None:
                return rtn_lin(name, w, steps[name])
            col, st = None, steps[name]
            if col_alpha > 0:
                col = ecq.column_factors(H, col_alpha, col_clamp)
                st = ecq.floor_step(w, st, col, max_code)
            act = os.environ.get("QUANT_ECQ_ACT_ORDER", "1") == "1"
            if use_dq and not is_embed_like(name):
                d = dq_step(w, st, col)
                kw = {}
                if dq_lam > 0:  # rate model: per (row class, state) tables of the trellis RTN codes on the same grid
                    bits, cls = dq.rate_model(w, d, H.device, dq_rate_cols)
                    kw = {"lam": dq_lam, "bits": bits, "cls": cls}
                return dq.gptq_dq(w, H, d, damp=damp, blocksize=dq_block, act_order=dq_order if act else False, col=col, **kw)
            return ecq.gptq_u(w, H, st, damp=damp, act_order=act, col=col, rdo=rdo)

        def group_quantizer(names, ws, H):
            # stacked rows of linears sharing an input; per-row steps make the stack exact (no column factors here)
            if use_dq and col_alpha <= 0 and not any(is_embed_like(n) for n in names):
                # one trellis search over the stacked rows (GPTQ is row-separable: same result as one call per tensor,
                # with a third / half of the per-column steps for q/k/v and gate/up)
                sizes = [w.shape[0] for w in ws]
                wcat = torch.cat(ws, 0)
                d = torch.cat([dq_step(w, steps[n]) for n, w in zip(names, ws)])
                kw = {}
                if dq_lam > 0:
                    bits, cls = dq.rate_model(wcat, d, H.device, dq_rate_cols)
                    kw = {"lam": dq_lam, "bits": bits, "cls": cls}
                u = dq.gptq_dq(wcat, H, d, damp=damp, blocksize=dq_block, act_order=dq_order, split=sizes, **kw)
                out, r0 = [], 0
                for i, sz in enumerate(sizes):
                    out.append(ecq.UResult(u.codes[r0 : r0 + sz].contiguous(), u.step[r0 : r0 + sz].contiguous(), dq=True, perm=u.perm, okey=u.okey,
                                           chunk=u.chunk, est=u.est[i]))
                    r0 += sz
                return out
            if col_alpha > 0 or use_dq:
                return [quantizer(n, w, H) for n, w in zip(names, ws)]
            sizes = [w.shape[0] for w in ws]
            u = ecq.gptq_u(torch.cat(ws, 0), H, torch.cat([steps[n] for n in names]), damp=damp,
                           act_order=os.environ.get("QUANT_ECQ_ACT_ORDER", "1") == "1", rdo=rdo)
            out, r0 = [], 0
            for sz in sizes:
                out.append(ecq.UResult(u.codes[r0 : r0 + sz].contiguous(), u.step[r0 : r0 + sz].contiguous()))
                r0 += sz
            return out

        lin = quantize_model_gptq(
            model, samples, QSpec(bits=8, group=1), damp=damp, true_sequential=True, time_left=lambda: deadline.left() - 300, quantizer=quantizer,
            group_quantizer=group_quantizer if (use_dq or os.environ.get("QUANT_ECQ_GROUP_GPTQ", "0") == "1") else None,
        )
        lin = {n: q for n, q in lin.items() if n in base.shapes and not is_embed_like(n)}
        for n in names_2d:
            if n not in lin:
                lin[n] = q_emb(n, steps[n])
        set_weights(model, lin, 0)
        r, sigma = measure()
        if emb_scan and k == 0 and embed_names:
            # Lever 1c (QUANT_FF_EMB_SCAN=1): with probe 0's block codes in place, re-quantise the embeddings (RTN) at
            # c0 x f for a few f and pick f minimising ln(bytes) - k_blk * share_blocks * (r(1) - r(f)): a lower r here lets
            # the blocks land coarser at the block bytes slope k_blk. Measured 25 Nov: t05 f 0.8 -> r -0.0062 for +1.6 MB
            # (~0.26 MB per 0.001 r, blocks ~2 MB per 0.001 r), t23 f0.8 -14%, t15 f0.8 worse: the right f is task-specific.
            t_sc = time.time()
            k_blk = float(os.environ.get("QUANT_FF_EMB_SCAN_K", 8.0))
            c_e0 = alloc.fixed_c[embed_names[0]]

            def emb_bytes(qd):
                _, fr = ffcodec.encode([ffcodec.Entry(n, base.shapes[n], q=qd[n]) for n in qd], level=int(os.environ.get("QUANT_FF_ZSTD", 0)))
                return ffcodec.encoded_size(fr)
            rest = 0
            for n, q in lin.items():
                if n not in embed_names:
                    e_ = getattr(q, "est", None)
                    rest += e_ if isinstance(e_, int) else int(q.codes.numel())
            emb_now = {n: lin[n] for n in embed_names if n in lin}
            scan = {1.0: (r, sigma, emb_now, emb_bytes(emb_now))}
            for f_ in [float(x) for x in os.environ.get("QUANT_FF_EMB_SCAN_F", "0.35,0.5,0.65,0.8,1.25,1.6,2.0").split(",") if x]:
                alloc.fixed_c = {n: c_e0 * f_ for n in embed_names}
                st_e = alloc.steps(c)
                q_e = {n: q_emb(n, st_e[n]) for n in emb_now}
                set_weights(model, q_e, 0)
                r_f, s_f = measure()
                scan[f_] = (r_f, s_f, q_e, emb_bytes(q_e))
            share = rest / max(1, rest + scan[1.0][3])

            def score(f_):
                return math.log(rest + scan[f_][3]) - k_blk * share * (scan[1.0][0] - scan[f_][0])
            f_best = min(scan, key=score)
            log.info("EMBSCAN (k_blk %.1f, blocks ~%.1f MB): %s -> f %.3g (%.0fs)", k_blk, rest / 1e6,
                     "; ".join(f"f {f_:g}: r {scan[f_][0]:.4f} emb {scan[f_][3] / 1e6:.2f} MB score {score(f_):.4f}" for f_ in sorted(scan)), f_best, time.time() - t_sc)
            alloc.fixed_c = {n: c_e0 * f_best for n in embed_names}
            steps = alloc.steps(c)
            r, sigma, q_e, _ = scan[f_best]
            lin.update(q_e)
            set_weights(model, q_e, 0)
            del scan
            # row-class step scans on the embeddings at f_best (one eval per option). Bytes: the smaller of the zstd frame
            # (ffcodec: wins while trimmed rows repeat) and the row-class rANS estimate, as the final coder will choose.
            from quant import rans as _rans

            def emb_bytes_rans(qd):
                tot = 0
                for n_, q_ in qd.items():
                    cd = q_.codes.to(torch.int8).contiguous().cpu().numpy()
                    spread = np.abs(cd.astype(np.float32)).mean(1)
                    key = np.clip(np.round(16.0 * np.log2(spread + 1e-3)) + 128, 0, 255).astype(np.uint8)
                    pm = np.argsort(key, kind="stable")
                    tot += _rans.estimate_bytes(np.ascontiguousarray(cd[pm]), key[pm]) + cd.shape[0]
                return tot

            def emb_bytes_best(qd):
                return min(emb_bytes(qd), emb_bytes_rans(qd)) if trimmed else emb_bytes_rans(qd)

            def row_scan(tag, options, apply, cur):
                """options: labels (the first = the current state, already quantised as cur = (r, sigma, q_e)); apply(label)
                sets the allocation / overrides for that option. Keeps the option with the best block-equivalent score."""
                t_s = time.time()
                first = options[0]
                sc_ = {first: (cur[0], cur[1], cur[2], emb_bytes_best(cur[2]))}
                for o_ in options[1:]:
                    apply(o_)
                    st_o = alloc.steps(c)
                    q_o = {n_: q_emb(n_, st_o[n_]) for n_ in cur[2]}
                    set_weights(model, q_o, 0)
                    r_o, s_o = measure()
                    sc_[o_] = (r_o, s_o, q_o, emb_bytes_best(q_o))

                def score_(o_):
                    return math.log(rest + sc_[o_][3]) - k_blk * share * (sc_[first][0] - sc_[o_][0])
                best_ = min(sc_, key=score_)
                log.info("%s (f %.3g, k_blk %.1f): %s -> %s (%.0fs)", tag, f_best, k_blk,
                         "; ".join(f"{o_}: r {sc_[o_][0]:.4f} emb {sc_[o_][3] / 1e6:.2f} MB score {score_(o_):.4f}" for o_ in options), best_, time.time() - t_s)
                apply(best_)
                set_weights(model, sc_[best_][2], 0)
                return sc_[best_][0], sc_[best_][1], sc_[best_][2]

            if unseen and deadline.left() > 1200 + dq_reserve:
                # lever E1: unseen-row step factor U at f_best (the trim was rejected)
                def set_u(u_):
                    for n_, m_ in unseen.items():
                        alloc.base[n_] = unseen_base0[n_] * torch.where(m_, torch.tensor(float(u_)), torch.tensor(1.0))
                        if n_ in rare_base0:
                            rare_base0[n_] = alloc.base[n_].clone()
                u_opts = [1.0] + [float(x) for x in os.environ.get("QUANT_FF_EMB_UNSEEN_U", "1.5,2,3,4,6").split(",") if x]
                r, sigma, q_e = row_scan("EMBUNSEEN (%s rows)" % ",".join(str(int(m_.sum())) for m_ in unseen.values()), u_opts, set_u, (r, sigma, q_e))
            soft_names = [n_ for n_ in trimmed if n_ in emb_gptq and n_ in q_e]
            if soft_names and os.environ.get("QUANT_FF_EMB_SOFT", "1") == "1" and deadline.left() > 1200 + dq_reserve:
                # soft trim (the trim was kept): the trimmed rows of the head table from their original values, GPTQ against
                # head_H, at step = S x the rms of those rows; "trim" (the mean row) stays an option
                gp0 = {n_: emb_gptq[n_].clone() for n_ in soft_names}
                c_emb = {n_: alloc.fixed_c[n_] for n_ in soft_names}
                orig = {}
                for n_ in soft_names:
                    with safe_open(str(base.where[n_]), framework="pt") as sf:
                        orig[n_] = float(sf.get_tensor(n_)[trimmed[n_]].float().pow(2).mean().sqrt())

                def set_soft(o_):
                    for n_ in soft_names:
                        ts, m_ = trim_state[n_], trimmed[n_]
                        if o_ == "trim":
                            base.overrides[n_] = ts["over"]
                            alloc.base[n_] = ts["base"].clone()
                            if ts["floor"] is not None:
                                alloc.floor[n_] = ts["floor"].clone()
                            emb_gptq[n_] = gp0[n_].clone()
                        else:
                            base.overrides.pop(n_, None)
                            alloc.base[n_] = torch.where(m_, torch.tensor(float(o_) * orig[n_] / c_emb[n_]), ts["base"])
                            if ts["floor0"] is not None:
                                alloc.floor[n_] = ts["floor0"].clone()
                            emb_gptq[n_] = gp0[n_] | m_
                        if n_ in rare_base0:
                            rare_base0[n_] = alloc.base[n_].clone()
                s_opts = ["trim"] + [float(x) for x in os.environ.get("QUANT_FF_EMB_SOFT_S", "2,2.8,4").split(",") if x]
                r, sigma, q_e = row_scan("EMBSOFT (%s rows)" % ",".join(str(int(trimmed[n_].sum())) for n_ in soft_names), s_opts, set_soft, (r, sigma, q_e))
            if rare and any(bool(m_.any()) for m_ in rare.values()) and os.environ.get("QUANT_FF_EMB_RARE", "1") == "1" and deadline.left() > 1200 + dq_reserve:
                # rare-row step factor (GPTQ rows): lever E1's idea for the rows seen 1..QUANT_FF_EMB_RARE_MAX times
                def set_rare(x_):
                    for n_, m_ in rare.items():
                        alloc.base[n_] = rare_base0[n_] * torch.where(m_, torch.tensor(float(x_)), torch.tensor(1.0))
                r_opts = [1.0] + [float(x) for x in os.environ.get("QUANT_FF_EMB_RARE_X", "2,3,4").split(",") if x]
                r, sigma, q_e = row_scan("EMBRARE (%s rows)" % ",".join(str(int(m_.sum())) for m_ in rare.values()), r_opts, set_rare, (r, sigma, q_e))
            steps = alloc.steps(c)
            lin.update(q_e)
            set_weights(model, q_e, 0)
        loss = r * ref_loss
        history.append((c, r))
        entries = []
        for name, shape in base.shapes.items():
            if name in lin:
                entries.append(ffcodec.Entry(name, shape, q=lin[name]))
            else:
                entries.append(ffcodec.Entry(name, shape, tensor=base.tensor(name)))
        clipped = sum(int((q.codes.abs() == ecq.QMAX).sum()) for q in lin.values())
        log.info("ECQ probe %d: c %.5g, eval loss %.5f r %.4f (sigma %.4f), clipped codes %d, %.0fs", k, c, loss, r, sigma, clipped, time.time() - t0)
        if tgt is None:
            consider(f"ecq-{alloc.kind}-c{c:.4g}", entries, r + margin)
        else:
            records, frames, est = consider(f"ecq-{alloc.kind}-c{c:.4g}", entries, r + bias, sigma=sigma)
            sc = tgt.expected_ln_s(est, 1.0 + (r - 1.0) * (1.0 - tune_f) + bias, sigma)
            if final is None or sc < final["score"]:
                final = {"lin": lin, "r": r, "name": f"ecq-{alloc.kind}-c{c:.4g}", "score": sc, "steps": steps}
            size_pts.append((r, est))
            sigma_now = sigma
            # the aim's slope is clamped to [-QUANT_FF_AIM_SLOPE_MAX, -2] (default 8 = the prior slope, aim ~1.0396 r_cal: real
            # data is ~2x more r-sensitive than the CNN proxy (T1), so the aim stays at ~1.039-1.040 as in 2bfa93f): with the
            # embeddings held, two block probes 1.25x apart can differ by ~0.001 r (probe noise), and the fitted slope (-40 on
            # t23, 25 Nov 16:15) put the aim at r 1.047, past the 1.046 guard (G3)
            aim_slope = max(-float(os.environ.get("QUANT_FF_AIM_SLOPE_MAX", 8.0)), landing.bytes_slope(size_pts))
            aim = pre(tgt.aim(sigma, aim_slope / (1.0 - tune_f)) - bias)
            log.info("landing aim r_cal %.4f (sigma %.4f, slope %.2f, used %.2f)", aim, sigma, landing.bytes_slope(size_pts), aim_slope)
            cand = {"c": c, "r": r, "sigma": sigma, "lin": lin, "enc": split_encoded(records, frames), "steps": steps}
            side = "lo" if r <= aim else "hi"
            other = "hi" if side == "lo" else "lo"
            cur = keep.get(side)
            if cur is None or (side == "lo" and r > cur["r"]) or (side == "hi" and r < cur["r"]):
                keep[side] = cand
            # a kept probe on the wrong side of the moved aim changes sides
            for sd in ("lo", "hi"):
                kp = keep.get(sd)
                if kp is not None and ((sd == "lo" and kp["r"] > aim) or (sd == "hi" and kp["r"] <= aim)):
                    keep.pop(sd)
                    alt = "hi" if sd == "lo" else "lo"
                    if alt not in keep or (alt == "lo" and kp["r"] > keep[alt]["r"]) or (alt == "hi" and kp["r"] < keep[alt]["r"]):
                        keep[alt] = kp
            del records, frames, cand
        probe_time = time.time() - t0
        del entries, lin
        gc.collect()
        if realloc and not realloc_done and tgt is not None and "lo" in keep and "hi" in keep and \
                deadline.left() > 28 * 25 + 2 * (probe_time or 300) * 1.2 + 330 + dq_reserve:
            realloc_done = True
            try:
                changed = block_realloc(keep["lo"], keep["hi"])
            except Exception as exc:  # noqa: BLE001
                log.warning("block realloc failed: %r", exc)
                changed = False
            if changed:
                c = math.sqrt(keep["lo"]["c"] * keep["hi"]["c"])
                keep.clear()
                history.clear()
                gc.collect()
                continue
        if abs(r - aim) <= tol:
            break
        if tgt is not None and "lo" in keep and "hi" in keep and keep["hi"]["c"] / keep["lo"]["c"] <= float(os.environ.get("QUANT_FF_MIX_WIDTH", 1.3)):
            log.info("ECQ: bracket c %.4g / %.4g (r %.4f / %.4f) narrow enough: mixing", keep["lo"]["c"], keep["hi"]["c"], keep["lo"]["r"], keep["hi"]["r"])
            break
        nb = bracket_next(c) if tgt is not None else None
        # secant in log space: log(r - 1) = a log c + b (a ~ 2 for uniform noise); one point -> a = 2
        c = nb if nb is not None else next_gptq_c(history, rtn_hist, aim, c, r)

    # Layer mix (landing mode): with a probe under the aim (lo) and one over it (hi), take the first m mixing units
    # (decoder blocks, embeddings last) from hi and the rest from lo. r and bytes move in ~1/units steps between the two
    # probes, so the landing gets much finer than a GPTQ probe can be placed, at the cost of one eval pass per step.
    if tgt is not None and "lo" in keep and "hi" in keep and deadline.left() > 240 + dq_reserve:
        lo, hi = keep["lo"], keep["hi"]
        # decoder blocks first, embeddings / head last: on t15 the tied embedding alone moved r +0.0045 for 4 MB
        # QUANT_FF_MIX_UNIT=linear (default): every decoder Linear is its own unit (block order, then module order), so
        # one unit moves r by ~1/200 of the bracket instead of ~1/29; "block": the 2bfa93f rule (one unit per block)
        mix_linear = os.environ.get("QUANT_FF_MIX_UNIT", "linear") == "linear"
        pos = {n: i for i, n in enumerate(names_2d)}

        def ukey(n: str):
            return (unit_of(n), pos[n]) if mix_linear else unit_of(n)
        blocks = [n for n in names_2d if unit_of(n) >= 0]
        units = sorted({ukey(n) for n in blocks}) + sorted({ukey(n) for n in names_2d if unit_of(n) < 0})
        unit_of_name = {n: ukey(n) for n in names_2d}
        # lever 2 (QUANT_FF_MIX_ORDER=gain): measure each block's swap lo -> hi alone (one eval per block on the paired
        # calibration rows) and mix the blocks in order of bytes saved per unit r (knapsack order), not block order
        order_gain = os.environ.get("QUANT_FF_MIX_ORDER", "block") == "gain"
        meas = {}
        if os.environ.get("QUANT_FF_UNIT_SWAP", "0") == "1" or order_gain:
            try:
                meas = unit_swaps(lo, hi)
            except Exception as exc:  # noqa: BLE001 - instrumentation only: the mix below still runs
                log.warning("UNITSWAP failed: %r", exc)
        if order_gain and all(f"block{b}" in meas for b in {unit_of(n) for n in blocks}):
            def eff(b):
                dr, dB = meas[f"block{b}"]
                return float("inf") if dr <= 0 else dB / dr
            rank = {b: i for i, b in enumerate(sorted({unit_of(n) for n in blocks}, key=lambda b: -eff(b)))}
            units = sorted({ukey(n) for n in blocks}, key=lambda u: (rank[u[0] if isinstance(u, tuple) else u], u)) + \
                sorted({ukey(n) for n in names_2d if unit_of(n) < 0})
            log.info("ECQ mix order by measured gain: %s", ",".join(str(b) for b in sorted(rank, key=rank.get)))
        m_lo, r_lo, m_hi, r_hi = 0, lo["r"], len(units), hi["r"]
        tol_mix = float(os.environ.get("QUANT_FF_MIX_TOL", 0.0008))
        eval_time = None
        for j in range(int(os.environ.get("QUANT_FF_MIX_EVALS", 8 if mix_linear else 5))):
            if m_hi - m_lo <= 1 or (eval_time is not None and deadline.left() < eval_time * 2 + 180 + dq_reserve) or deadline.left() < 240 + dq_reserve:
                break
            t0 = time.time()
            frac = (aim - r_lo) / max(1e-9, r_hi - r_lo)
            m = int(round(m_lo + frac * (m_hi - m_lo)))
            m = min(m_hi - 1, max(m_lo + 1, m))
            from_hi = set(units[:m])
            src_of = {n: (hi if unit_of_name[n] in from_hi else lo) for n in names_2d}
            set_weights(model, {n: src_of[n]["lin"][n] for n in names_2d if n in params}, 0)
            r, sigma = measure()
            parts = [(src_of[name] if name in src_of else lo)["enc"][name] for name in base.shapes]
            enc = join_encoded(parts)
            mname = f"ecq-mix-c{lo['c']:.4g}-c{hi['c']:.4g}-m{m}of{len(units)}"
            mix_entries = [ffcodec.Entry(name, shape, q=src_of[name]["lin"][name]) if name in src_of else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                           for name, shape in base.shapes.items()]
            _, _, est = consider(mname, mix_entries, r + bias, sigma=sigma, enc=enc)
            del mix_entries
            sc = tgt.expected_ln_s(est, 1.0 + (r - 1.0) * (1.0 - tune_f) + bias, sigma)
            if final is None or sc < final["score"]:
                final = {"lin": {n: src_of[n]["lin"][n] for n in names_2d}, "r": r, "name": mname, "score": sc,
                         "steps": {n: src_of[n]["steps"][n] for n in names_2d}}
            size_pts.append((r, est))
            eval_time = time.time() - t0
            log.info("ECQ mix %d: %d/%d units from c %.4g, r %.4f (sigma %.4f), ~%.1f MB, %.0fs", j, m, len(units), hi["c"], r, sigma, est / 1e6, eval_time)
            del parts, enc
            if r <= aim:
                m_lo, r_lo = m, r
            else:
                m_hi, r_hi = m, r
            if abs(r - aim) <= tol_mix:
                break
        # lever 2c v2 (QUANT_FF_PICK_OVER > 0 with QUANT_FF_BLOCK_PICK=1): a block pick lowers r at the same bytes (t23, 25 Nov:
        # held-out rows -0.0099, test r 1.0286 vs 1.0382 predicted for the unpicked model), which only pays if the blocks
        # were coarser to begin with: build a second mix landing QUANT_FF_PICK_OVER above the aim and pick on THAT; the
        # picked model competes with the normal landing in consider() (expected S), so a weak pick leaves the landing as is.
        over = float(os.environ.get("QUANT_FF_PICK_OVER", 0))
        if over > 0 and os.environ.get("QUANT_FF_BLOCK_PICK", "0") == "1" and probe_time is not None and \
                deadline.left() > probe_time * 1.25 + 30 * 0.6 * (eval_t or 25) + 360 + dq_reserve:
            r_t = aim + over
            if hi["r"] <= r_t:
                m = len(units)
            else:
                frac = (r_t - r_lo) / max(1e-9, hi["r"] - r_lo)
                m = min(len(units), max(m_lo + 1, int(round(m_lo + frac * (len(units) - m_lo)))))
            from_hi = set(units[:m])
            src_of = {n: (hi if unit_of_name[n] in from_hi else lo) for n in names_2d}
            lin_o = {n: src_of[n]["lin"][n] for n in names_2d}
            set_weights(model, {n: lin_o[n] for n in names_2d if n in params}, 0)
            r_o, _ = measure()
            log.info("PICKOVER: over-aim mix %d/%d units, r %.4f (target %.4f)", m, len(units), r_o, r_t)
            fin_over = {"lin": lin_o, "r": r_o, "name": f"ecq-over-c{lo['c']:.4g}-c{hi['c']:.4g}-m{m}of{len(units)}",
                        "steps": {n: src_of[n]["steps"][n] for n in names_2d}}
            res_p = None
            try:
                res_p = block_pick(fin_over)
                picked_over = True
            except Exception as exc:  # noqa: BLE001
                log.warning("pick over failed: %r", exc)
            # lever 2c v3 (QUANT_FF_PICK_RELAND=1): the pick lowers r (t15 25 Nov: 1.0351 -> 1.0262 cal, 1.0246 test) but the
            # over-aim mix can only reach the hi probe; re-land: one GPTQ probe coarser than hi (secant, slope prior 2, at most
            # 1.5x), then a linear-unit mix between the picked model (lo side) and that probe, each mix point a candidate
            if res_p is not None and os.environ.get("QUANT_FF_PICK_RELAND", "0") == "1":
                comb_p, r_p = res_p
                if r_p < aim - 0.002 and deadline.left() > (probe_time or 300) * 1.25 + 6 * 45 + 330 + dq_reserve:
                    try:
                        t_rl = time.time()
                        c_ref = hi["c"]
                        c_new = c_ref * math.exp((math.log(max(aim + 0.004 - 1.0, 1e-5)) - math.log(max(r_p - 1.0, 1e-5))) / 2.0)
                        c_new = min(c_ref * 1.5, max(c_ref * 1.05, c_new))
                        st_new = alloc.steps(c_new)
                        lin_h = gptq_at(st_new, samples)
                        for n in names_2d:
                            if n not in lin_h:
                                lin_h[n] = comb_p[n] if is_embed_like(n) else ecq.rtn_u(base.tensor(n), st_new[n], device=device)
                        set_weights(model, {n: lin_h[n] for n in names_2d if n in params}, 0)
                        r_h, s_h = measure()
                        ents = [ffcodec.Entry(name, shape, q=lin_h[name]) if name in lin_h else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                                for name, shape in base.shapes.items()]
                        consider(f"ecq-reland-c{c_new:.4g}", ents, r_h + bias, sigma=s_h)
                        del ents
                        log.info("PICKRELAND: probe c %.4g r %.4f (picked r %.4f, aim %.4f), %.0fs", c_new, r_h, r_p, aim, time.time() - t_rl)
                        if r_h > aim:
                            u_lo, u_rlo, u_hi, u_rhi = 0, r_p, len(units), r_h
                            for j in range(6):
                                if u_hi - u_lo <= 1 or deadline.left() < 240 + dq_reserve:
                                    break
                                frac = (aim - u_rlo) / max(1e-9, u_rhi - u_rlo)
                                m2 = min(u_hi - 1, max(u_lo + 1, int(round(u_lo + frac * (u_hi - u_lo)))))
                                fh = set(units[:m2])
                                mixl = {n: (lin_h[n] if unit_of_name[n] in fh else comb_p[n]) for n in names_2d}
                                set_weights(model, {n: mixl[n] for n in names_2d if n in params}, 0)
                                r_m, s_m = measure()
                                ents = [ffcodec.Entry(name, shape, q=mixl[name]) if name in mixl else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                                        for name, shape in base.shapes.items()]
                                _, _, est_m = consider(f"ecq-reland-mix-m{m2}of{len(units)}", ents, r_m + bias, sigma=s_m)
                                del ents, mixl
                                log.info("PICKRELAND mix %d: %d/%d units from the coarse probe, r %.4f, ~%.1f MB", j, m2, len(units), r_m, est_m / 1e6)
                                if r_m <= aim:
                                    u_lo, u_rlo = m2, r_m
                                else:
                                    u_hi, u_rhi = m2, r_m
                                if abs(r_m - aim) <= tol_mix:
                                    break
                        del lin_h
                    except Exception as exc:  # noqa: BLE001
                        log.warning("pick re-land failed: %r", exc)
            del fin_over, lin_o
    # Lever 2c v4 (land6, QUANT_FF_PICK4=1): the per-block best-of-two-draws pick lowers r by ~0.007-0.017 at the SAME bytes
    # (25 Nov, 4 of 5 picks accepted by the held-out odd rows and confirmed on the test rows), but v1-v3 could not turn it
    # into bytes: the picked model landed far under the aim (t15 r_test 1.0246) with no time or no coarser probe to re-land.
    # v4: (1) make sure a probe sits at least QUANT_FF_PICK4_OVER above the aim (one extra probe if the search ended
    # without one), (2) build the mix that lands `over` above the aim, draw B at its steps, pick per block on the even
    # rows, accept on the odd rows, (3) re-land the picked model with unit swaps that cost no new probe: toward the coarse
    # probe's codes while it is under the aim, back to the fine probe's codes if it is over. Every step is a candidate in
    # consider() (expected S), r carries the selection correction, and without time or without acceptance the normal
    # landing stands.
    if os.environ.get("QUANT_FF_PICK4", "1") == "1" and tgt is not None and final is not None and probe_time is not None and keep.get("lo") is not None:
        over = float(os.environ.get("QUANT_FF_PICK4_OVER", 0.006))
        ev_t = eval_t or 25.0
        need = probe_time * 1.25 + 30 * 0.6 * ev_t + 10 * ev_t + 330 + dq_reserve
        t_p4 = time.time()
        try:
            lo, hi = keep["lo"], keep.get("hi")
            if (hi is None or hi["r"] < aim + over) and deadline.left() > need + probe_time * 1.25:
                c_ref, r_ref = (hi["c"], hi["r"]) if hi is not None else (lo["c"], lo["r"])
                c_new = c_ref * math.exp((math.log(max(aim + over + 0.003 - 1.0, 1e-5)) - math.log(max(r_ref - 1.0, 1e-5))) / 2.0)
                c_new = min(c_ref * 1.3, max(c_ref * 1.06, c_new))
                st_new = alloc.steps(c_new)
                lin_h = gptq_at(st_new, samples)
                for n in names_2d:
                    if n not in lin_h:
                        lin_h[n] = q_emb(n, st_new[n]) if is_embed_like(n) else ecq.rtn_u(base.tensor(n), st_new[n], device=device)
                set_weights(model, {n: lin_h[n] for n in names_2d if n in params}, 0)
                r_h, s_h = measure()
                ents = [ffcodec.Entry(name, shape, q=lin_h[name]) if name in lin_h else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                        for name, shape in base.shapes.items()]
                rec_h, fr_h, est_h = consider(f"ecq-{alloc.kind}-c{c_new:.4g}", ents, r_h + bias, sigma=s_h)
                del ents
                history.append((c_new, r_h))
                log.info("PICK4: coarse probe c %.4g r %.4f (aim %.4f + %.4f), ~%.1f MB, %.0fs", c_new, r_h, aim, over, est_h / 1e6, time.time() - t_p4)
                cand_h = {"c": c_new, "r": r_h, "sigma": s_h, "lin": lin_h, "enc": split_encoded(rec_h, fr_h), "steps": st_new}
                if r_h > aim and (hi is None or r_h > hi["r"]):
                    hi = cand_h
                elif r_h <= aim and r_h > lo["r"]:
                    lo = cand_h
                del rec_h, fr_h
            if hi is not None and hi["r"] > aim and deadline.left() > need:
                pos4 = {n: i for i, n in enumerate(names_2d)}
                blocks4 = [n for n in names_2d if unit_of(n) >= 0]
                units4 = sorted({(unit_of(n), pos4[n]) for n in blocks4})
                uo4 = {n: (unit_of(n), pos4[n]) for n in names_2d}
                r_t = min(aim + over, hi["r"])
                frac = (r_t - lo["r"]) / max(1e-9, hi["r"] - lo["r"])
                m0 = min(len(units4), max(1, int(round(frac * len(units4)))))
                from_hi = set(units4[:m0])
                src0 = {n: (hi if uo4[n] in from_hi else lo) for n in names_2d}
                lin_o = {n: src0[n]["lin"][n] for n in names_2d}
                set_weights(model, {n: lin_o[n] for n in names_2d if n in params}, 0)
                r_o, _ = measure()
                log.info("PICK4: over-aim mix %d/%d units from c %.4g, r %.4f (target %.4f; lo %.4f hi %.4f)", m0, len(units4), hi["c"], r_o, r_t, lo["r"], hi["r"])
                P = block_pick({"lin": lin_o, "r": r_o, "name": f"ecq-over-c{lo['c']:.4g}-c{hi['c']:.4g}-m{m0}of{len(units4)}",
                                "steps": {n: src0[n]["steps"][n] for n in names_2d}}, full=True)
                if P is not None:
                    r_p = P["r"] + P["corr"]
                    log.info("PICK4: picked %d blocks, r %.4f (+%.4f selection correction; odd-row gain %.4f, even %.4f), ~%.1f MB", len(P["pick"]),
                             P["r"], P["corr"], P["gain_odd"], P["gain_even"], P["est"] / 1e6)

                    def p4_consider(tag, lin_m, parts):
                        set_weights(model, {n: lin_m[n] for n in names_2d if n in params}, 0)
                        r_m, s_m = measure()
                        ents_m = [ffcodec.Entry(name, shape, q=lin_m[name]) if name in lin_m else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                                  for name, shape in base.shapes.items()]
                        _, _, est_m = consider(tag, ents_m, r_m + P["corr"] + bias, sigma=s_m, enc=join_encoded(parts))
                        return r_m + P["corr"], s_m, est_m

                    def p4_final(tag, lin_m, r_m, s_m, est_m):
                        nonlocal final
                        sc_m = tgt.expected_ln_s(est_m, r_m + bias, s_m)
                        if sc_m < final["score"]:
                            final = {"lin": dict(lin_m), "r": r_m, "name": tag, "score": sc_m, "steps": {n: src0[n]["steps"][n] for n in names_2d},
                                     "corr": P["corr"]}
                    p4_final(P["name"], P["lin"], r_p, P["sigma"], P["est"])
                    tol_p = float(os.environ.get("QUANT_FF_MIX_TOL", 0.0008))
                    if r_p < aim - tol_p:
                        swap, other, tagk = [u for u in units4 if u not in from_hi], hi, "up"  # still on the fine probe's steps -> coarse codes
                    elif r_p > aim + tol_p:
                        swap, other, tagk = [u for u in reversed(units4) if u in from_hi], lo, "down"  # coarse steps -> fine codes
                    else:
                        swap, other, tagk = [], None, ""
                    j_lo, j_hi, r_a, r_b = 0, len(swap), r_p, None
                    for it in range(int(os.environ.get("QUANT_FF_PICK4_EVALS", 6))):
                        if j_hi - j_lo <= 1 and it > 0 or not swap or deadline.left() < 2 * ev_t + 240 + dq_reserve:
                            break
                        if it == 0:
                            j = len(swap)  # the far end first: is the aim inside?
                        else:
                            frac = (aim - r_a) / (r_b - r_a) if r_b is not None and abs(r_b - r_a) > 1e-9 else 0.5
                            j = min(j_hi - 1, max(j_lo + 1, int(round(j_lo + frac * (j_hi - j_lo)))))
                        sw = set(swap[:j])
                        lin_m = {n: (other["lin"][n] if uo4[n] in sw else P["lin"][n]) for n in names_2d}
                        parts = [((other if uo4[name] in sw else P) if name in uo4 else lo)["enc"][name] for name in base.shapes]
                        tag = f"{P['name']}-{tagk}{j}of{len(swap)}"
                        r_m, s_m, est_m = p4_consider(tag, lin_m, parts)
                        log.info("PICK4 re-land %d (%s): %d/%d units swapped, r %.4f (aim %.4f), ~%.1f MB", it, tagk, j, len(swap), r_m, aim, est_m / 1e6)
                        p4_final(tag, lin_m, r_m, s_m, est_m)
                        inside = (r_m >= aim) if tagk == "up" else (r_m <= aim)
                        if it == 0 and not inside:
                            break  # even the full swap stays on the same side of the aim: it is the best this bracket can do
                        if abs(r_m - aim) <= tol_p:
                            break
                        if (r_m < aim) == (tagk == "up"):
                            j_lo, r_a = j, r_m
                        else:
                            j_hi, r_b = j, r_m
                        del lin_m, parts
                del lin_o
            log.info("PICK4 done in %.0fs (%.0fs left)", time.time() - t_p4, deadline.left())
        except Exception as exc:  # noqa: BLE001 - the normal landing stands
            log.warning("PICK4 failed: %r", exc)
            import traceback
            log.warning("%s", traceback.format_exc())
    keep.clear()
    gc.collect()
    # Lever 1b (QUANT_FF_EMB_SEARCH=1): the embeddings were held at the first probe's c (lever 1); with the blocks landed,
    # re-quantise them (RTN, as always) at a few other constants and let the expected-S rule choose. t15 (real data, untrimmed
    # 72 MB embedding) paid for the held fine embedding (f0.8 +1.2%, wave-B candidate +~2% vs 2bfa93f), t23 / t05 gained
    # from a finer one (f0.8 -14% / -5%): the right constant is task-specific, so it is measured, one eval per option.
    if os.environ.get("QUANT_FF_EMB_SEARCH", "1") == "1" and tgt is not None and final is not None and emb_own_c \
            and getattr(alloc, "fixed_c", None) and deadline.left() > 300 + dq_reserve:
        c_e0 = next(iter(alloc.fixed_c.values()))
        facs = [float(x) for x in os.environ.get("QUANT_FF_EMB_SEARCH_F", "0.8,1.25,1.6").split(",") if x]
        names = [n for n in names_2d if n in params]
        set_weights(model, {n: final["lin"][n] for n in names}, 0)
        t_es = time.time()
        for f_ in facs:
            if deadline.left() < 240 + dq_reserve:
                log.info("EMBSEARCH: out of time")
                break
            t0 = time.time()
            alloc.fixed_c = {n: c_e0 * f_ for n in embed_names}
            st_e = alloc.steps(c_e0)
            emb_q = {n: q_emb(n, st_e[n]) for n in embed_names if n in names}
            set_weights(model, emb_q, 0)
            r_, sigma_ = measure()
            lin_e = dict(final["lin"])
            lin_e.update(emb_q)
            entries = [ffcodec.Entry(name, shape, q=lin_e[name]) if name in lin_e else ffcodec.Entry(name, shape, tensor=base.tensor(name))
                       for name, shape in base.shapes.items()]
            _, _, est = consider(final["name"] + f"-emb{f_:g}", entries, r_ + final.get("corr", 0.0) + bias, sigma=sigma_)
            log.info("EMBSEARCH factor %.3g (c %.4g): r %.4f, ~%.1f MB, %.0fs", f_, c_e0 * f_, r_, est / 1e6, time.time() - t0)
            del entries, lin_e
            set_weights(model, {n: final["lin"][n] for n in embed_names if n in names}, 0)
        alloc.fixed_c = {n: c_e0 for n in embed_names}
        log.info("EMBSEARCH done in %.0fs", time.time() - t_es)
        gc.collect()
    # Lever 2c (QUANT_FF_BLOCK_PICK=1): GPTQ outcomes at the same steps differ a lot (t23, 25 Nov: r 1.0381 vs 1.0267 at
    # c 1.042e-5 / 1.065e-5), while one model's calibration and test r agree within ~0.0025. A second GPTQ draw B at the
    # shipped candidate's exact steps (Hessians from the next QUANT_GPTQ_ROWS train rows) is made; each decoder block takes
    # B's codes when its single-block swap lowers r on the EVEN calibration rows; the combined model is accepted only if
    # its r on the ODD rows (not used for the choice) is lower than the candidate's there, and then competes in consider().
    if os.environ.get("QUANT_FF_BLOCK_PICK", "0") == "1" and not picked_over and tgt is not None and final is not None and final.get("steps") \
            and probe_time is not None and deadline.left() > probe_time * 1.25 + 30 * 0.6 * (eval_t or 25) + 330 + dq_reserve:
        try:
            block_pick(final)
        except Exception as exc:  # noqa: BLE001
            log.warning("block pick failed: %r", exc)
        gc.collect()
    # Optional tuning stage (QUANT_FF_TUNE=1): the search above aimed QUANT_FF_TUNE_F of the excess higher (see aim), so the
    # tuned candidate should land near the real aim; it is shipped only if its measured expected S is lower.
    if tgt is not None and tune_f > 0 and final is not None and deadline.left() > 420:
        t0 = time.time()
        lin_t = dict(final["lin"])
        budget = max(60.0, deadline.left() - float(os.environ.get("QUANT_FF_TUNE_RESERVE", 300)))
        raw = tune_candidate(model, lin_t, tune_train, tune_sel, budget, tok.pad_token_id if tok.pad_token_id is not None else 0, deadline.left)
        r, sigma = measure()
        log.info("ECQ tuned %s: r %.4f -> %.4f (sigma %.4f), %.0fs", final["name"], final["r"], r, sigma, time.time() - t0)
        entries = []
        for name, shape in base.shapes.items():
            if name in lin_t:
                entries.append(ffcodec.Entry(name, shape, q=lin_t[name]))
            else:
                entries.append(ffcodec.Entry(name, shape, tensor=raw.get(name, base.tensor(name))))
        consider(final["name"] + "-tuned", entries, r + bias, sigma=sigma)
    log.info("ECQ history (c, r): %s", ", ".join(f"({cc:.4g}, {rr:.4f})" for cc, rr in history))

    # Block-AP on the ECQ grid (EfficientQAT, Chen et al. 2024, arXiv:2407.11062): rounding learned per block against
    # the bf16 block outputs lowers r at a given step, so larger steps (fewer bytes) fit under the knee.
    if os.environ.get("QUANT_ECQ_BAP", "0") == "1" and history:
        from quant.blockap import blockap

        feasible = [cc for cc, rr in history if (rr <= aim if tgt is not None else rr + margin <= target["knee"])]
        c_b = (max(feasible) if feasible else min(history)[0]) * float(os.environ.get("QUANT_ECQ_BAP_GAIN", 1.15))
        bap_hist: list[tuple[float, float]] = []
        bap_time = None
        lin_names = [n for n in names_2d if not is_embed_like(n)]
        for k in range(int(os.environ.get("QUANT_ECQ_BAP_PROBES", 3))):
            left = deadline.left()
            if left < 600 or (bap_time is not None and left < bap_time * 1.25 + 180):
                log.info("ECQ Block-AP: no time for probe %d (%.0fs left)", k, left)
                break
            t0 = time.time()
            restore()
            steps = alloc.steps(c_b)
            init = {n: steps[n] for n in lin_names}
            res = blockap(
                model,
                pool,
                QSpec(bits=8, group=1),
                init,
                n_chunks=int(os.environ.get("QUANT_ECQ_BAP_CHUNKS", 512)),
                length=int(os.environ.get("QUANT_ECQ_BAP_LEN", 512)),
                batch=int(os.environ.get("QUANT_ECQ_BAP_BATCH", 4)),
                epochs=int(os.environ.get("QUANT_ECQ_BAP_EPOCHS", 2)),
                lr_w=float(os.environ.get("QUANT_ECQ_BAP_LR", 1e-4)),
                lr_q=1e-4,
                time_budget=max(60.0, deadline.left() - 300),
                fake_factory=lambda m, st: ecq.FakeQuantLinearU(m, st, learn_step=os.environ.get("QUANT_ECQ_BAP_LEARN_STEP", "0") == "1"),
            )
            if res is None:
                log.info("ECQ Block-AP: does not fit the budget")
                break
            lin = {n: q for n, q in res.items() if n in base.shapes}
            for n in names_2d:
                if n not in lin:
                    lin[n] = ecq.rtn_u(base.tensor(n), steps[n], device=device)
            set_weights(model, {n: lin[n] for n in names_2d if is_embed_like(n) or n not in res}, 0)
            r, sigma = measure()
            bap_hist.append((c_b, r))
            entries = [ffcodec.Entry(name, shape, q=lin[name]) if name in lin else ffcodec.Entry(name, shape, tensor=base.tensor(name)) for name, shape in base.shapes.items()]
            log.info("ECQ Block-AP probe %d: c %.5g r %.4f (%.0fs)", k, c_b, r, time.time() - t0)
            if tgt is None:
                consider(f"ecq-bap-c{c_b:.4g}", entries, r + margin)
            else:
                consider(f"ecq-bap-c{c_b:.4g}", entries, r + bias, sigma=sigma)
            bap_time = time.time() - t0
            del entries, lin, res
            gc.collect()
            torch.cuda.empty_cache()
            if abs(r - aim) <= tol:
                break
            c_b = secant(bap_hist, aim, c_b, r, 2.0)
        log.info("ECQ Block-AP history (c, r): %s", ", ".join(f"({cc:.4g}, {rr:.4f})" for cc, rr in bap_hist))
    del model
    gc.collect()
