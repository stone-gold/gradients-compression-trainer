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


def used_token_ids(task_id: str, dataset_arg: str, tok, extra_rows, time_limit: float) -> set[int] | None:
    """Token ids that occur anywhere in the task's train split (every string field of every row, tokenised in batches by
    the fast tokenizer), in the calibration rows and in the tokenizer's special / added tokens. None if the split cannot
    be read completely within time_limit (then nothing is trimmed)."""
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

    used: set[int] = set()
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
    used.update(int(i) for i in (tok.all_special_ids or []))
    used.update(int(i) for i in getattr(tok, "added_tokens_decoder", {}).keys())
    for t in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tok, t, None)
        if v is not None:
            used.add(int(v))
    return used


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
        if land == "expected":
            sg = sigma if sigma is not None else 0.003
            score = tgt.expected_ln_s(est, r_pred, sg)
            S = math.exp(min(score, 700.0)) if score < float("inf") else float("inf")
            log.info("candidate %s: ~%.1f MB, predicted r %.4f +- %.4f, P(r>knee) %.2f, expected S %.1f MB (best %s %.1f MB)", name, est / 1e6,
                     r_pred, sg, tgt.p_over(r_pred, sg, tgt.knee_r), S / 1e6, best["name"], best["S"] / 1e6)
            if (force or score < best["score"]) and deadline.left() > 60:
                info = {"candidate": name, "predicted_r": r_pred, "sigma": sg}
                total = ffcodec.write_artifact(out, args.task_id, records, frames, info)
                best.update(S=S, score=score, name=name, r=r_pred, bytes=total)
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
    if alloc_kind.startswith("fisher"):
        t0 = time.time()
        frows = [r for r in pool if r.n_scored > 0][-int(os.environ.get("QUANT_ECQ_FISHER_ROWS", 128)) :]
        fisher = ecq.fisher_rows(model, frows, names_2d, len(frows), int(os.environ.get("QUANT_ECQ_FISHER_LEN", 1024)))
        torch.cuda.empty_cache()
        log.info("Fisher in %.0fs", time.time() - t0)
    embed_row = os.environ.get("QUANT_ECQ_EMBED_ROW", "1") == "1"
    row_level = set() if alloc_kind != "fisher_row" else {n for n in names_2d if embed_row or n not in embed_names}
    clamp = {n: float(os.environ.get("QUANT_ECQ_EMBED_CLAMP" if n in embed_names else "QUANT_ECQ_ROW_CLAMP", 4.0)) for n in names_2d}
    alloc = ecq.Allocation(base.shapes, rms, fisher, row_level, clamp)
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
                    alloc.base[n] = torch.where(mask, shared.expand_as(alloc.base[n]), alloc.base[n])
                    if n in alloc.floor:
                        alloc.floor[n] = torch.where(mask, alloc.floor[n][mask].max().expand_as(alloc.floor[n]), alloc.floor[n])
            else:
                for n in plan:
                    params[n].data.copy_(base.tensor(n).to(params[n].dtype).to(params[n].device))

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
    history: list[tuple[float, float]] = []
    max_probes = int(os.environ.get("QUANT_ECQ_PROBES", 5))
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
        line = rtn_curve(rhist)
        if line is None:
            return secant(hist, target, c_now, r_now, 1.5)
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
        res = {n: ecq.rtn_u(params[n].data, steps[n], device=device) for n in names_2d if n in params}
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

    for k in range(max_probes):
        # GPTQ itself switches to RTN when less than ~1.2 x its time is left beyond a 300 s margin: never start a probe
        # that would degrade that way (it costs time and lands nowhere useful)
        if probe_time is not None and deadline.left() < probe_time * float(os.environ.get("QUANT_ECQ_GATE_MULT", 1.15)) + 330:
            log.info("ECQ: no time for another probe (%.0fs left, probe %.0fs)", deadline.left(), probe_time)
            break
        t0 = time.time()
        restore()
        steps = alloc.steps(c)

        def quantizer(name, w, H):
            if H is None:
                return ecq.rtn_u(w, steps[name], device=device)
            col, st = None, steps[name]
            if col_alpha > 0:
                col = ecq.column_factors(H, col_alpha, col_clamp)
                st = ecq.floor_step(w, st, col, max_code)
            return ecq.gptq_u(w, H, st, damp=damp, act_order=os.environ.get("QUANT_ECQ_ACT_ORDER", "1") == "1", col=col, rdo=rdo)

        def group_quantizer(names, ws, H):
            # stacked rows of linears sharing an input; per-row steps make the stack exact (no column factors here)
            if col_alpha > 0:
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
            group_quantizer=group_quantizer if os.environ.get("QUANT_ECQ_GROUP_GPTQ", "0") == "1" else None,
        )
        lin = {n: q for n, q in lin.items() if n in base.shapes and not is_embed_like(n)}
        for n in names_2d:
            if n not in lin:
                lin[n] = ecq.rtn_u(base.tensor(n), steps[n], device=device)
        set_weights(model, lin, 0)
        r, sigma = measure()
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
                final = {"lin": lin, "r": r, "name": f"ecq-{alloc.kind}-c{c:.4g}", "score": sc}
            size_pts.append((r, est))
            sigma_now = sigma
            aim = pre(tgt.aim(sigma, landing.bytes_slope(size_pts) / (1.0 - tune_f)) - bias)
            log.info("landing aim r_cal %.4f (sigma %.4f, slope %.2f)", aim, sigma, landing.bytes_slope(size_pts))
            cand = {"c": c, "r": r, "sigma": sigma, "lin": lin, "enc": split_encoded(records, frames)}
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
    if tgt is not None and "lo" in keep and "hi" in keep and deadline.left() > 240:
        lo, hi = keep["lo"], keep["hi"]
        # decoder blocks first, embeddings / head last: on t15 the tied embedding alone moved r +0.0045 for 4 MB
        units = sorted(u for u in {unit_of(n) for n in names_2d} if u >= 0) + ([-1] if any(unit_of(n) < 0 for n in names_2d) else [])
        m_lo, r_lo, m_hi, r_hi = 0, lo["r"], len(units), hi["r"]
        tol_mix = float(os.environ.get("QUANT_FF_MIX_TOL", 0.0008))
        eval_time = None
        for j in range(int(os.environ.get("QUANT_FF_MIX_EVALS", 5))):
            if m_hi - m_lo <= 1 or (eval_time is not None and deadline.left() < eval_time * 2 + 180) or deadline.left() < 240:
                break
            t0 = time.time()
            frac = (aim - r_lo) / max(1e-9, r_hi - r_lo)
            m = int(round(m_lo + frac * (m_hi - m_lo)))
            m = min(m_hi - 1, max(m_lo + 1, m))
            from_hi = set(units[:m])
            src_of = {n: (hi if unit_of(n) in from_hi else lo) for n in names_2d}
            set_weights(model, {n: src_of[n]["lin"][n] for n in names_2d if n in params}, 0)
            r, sigma = measure()
            parts = [(src_of[name] if name in src_of else lo)["enc"][name] for name in base.shapes]
            enc = join_encoded(parts)
            mname = f"ecq-mix-c{lo['c']:.4g}-c{hi['c']:.4g}-m{m}of{len(units)}"
            _, _, est = consider(mname, None, r + bias, sigma=sigma, enc=enc)
            sc = tgt.expected_ln_s(est, 1.0 + (r - 1.0) * (1.0 - tune_f) + bias, sigma)
            if final is None or sc < final["score"]:
                final = {"lin": {n: src_of[n]["lin"][n] for n in names_2d}, "r": r, "name": mname, "score": sc}
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
    keep.clear()
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
