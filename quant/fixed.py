"""Fixed track (QuantFixedTask): a compressed-tensors pack-quantized checkpoint in the task's exact format.

Pipeline, each stage replacing the previous checkpoint only when it is better on the calibration rows:
  1. RTN (clip-searched) straight from the input safetensors: a valid fallback within minutes.
  2. Sequential GPTQ (static groups, natural order) on calibration (+ train) rows.
  3. If time remains: end-to-end tuning of the bf16 group scales (integer codes and zero points frozen) on train rows.
quant_manifest.json is written after every accepted checkpoint, last.
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
from safetensors.torch import save_file

from quant import common
from quant.data import IGNORE_INDEX
from quant.data import Row
from quant.data import load_calibration
from quant.data import load_train_rows
from quant.packing import packed_tensors
from quant.quantize import QResult
from quant.quantize import QSpec
from quant.quantize import quantize_model_gptq
from quant.quantize import rtn


log = logging.getLogger("quant.fixed")
WEIGHTS_FILE = "model.safetensors"
EMBED_MARKERS = ("embed_tokens",)
LM_HEAD = "lm_head.weight"


def parse_spec(args) -> dict:
    raw = args.quant_spec or os.environ.get("QUANT_SPEC")
    if not raw:
        raise ValueError("fixed task without --quant-spec / QUANT_SPEC")
    spec = json.loads(raw)
    if spec.get("activation", "bf16") != "bf16":
        raise NotImplementedError(f"activation {spec.get('activation')} not supported")
    if spec.get("act_order"):
        raise NotImplementedError("act_order")
    return spec


def quantization_config(spec: dict) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "format": "pack-quantized",
                "weights": {
                    "type": "int",
                    "strategy": "group",
                    "num_bits": int(spec["weight_bits"]),
                    "group_size": int(spec["group_size"]),
                    "symmetric": bool(spec["symmetric"]),
                    "actorder": None,
                    "scale_dtype": "torch.bfloat16",
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }


def is_quantised_weight(name: str, shape) -> bool:
    return len(shape) == 2 and name.endswith(".weight") and not any(m in name for m in EMBED_MARKERS) and name != LM_HEAD


class InputCheckpoint:
    """The input model's safetensors: name -> (file, dtype, shape)."""

    def __init__(self, path: str):
        self.path = Path(path)
        files = sorted(self.path.glob("*.safetensors"))
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            wanted = set(json.loads(index.read_text())["weight_map"].values())
            files = [f for f in files if f.name in wanted]
        self.where: dict[str, Path] = {}
        self.shapes: dict[str, list[int]] = {}
        for f in files:
            with safe_open(str(f), framework="pt") as sf:
                for name in sf.keys():
                    self.where[name] = f
                    self.shapes[name] = list(sf.get_slice(name).get_shape())
        if not self.where:
            raise RuntimeError(f"no safetensors tensors under {path}")

    def tensor(self, name: str) -> torch.Tensor:
        with safe_open(str(self.where[name]), framework="pt") as sf:
            return sf.get_tensor(name)

    def quantised_names(self, group: int) -> list[str]:
        return [n for n, s in self.shapes.items() if is_quantised_weight(n, s)]


def write_checkpoint(
    out: Path,
    task_id: str,
    ckpt: InputCheckpoint,
    base_config: dict,
    qspec: QSpec,
    spec: dict,
    results: dict[str, QResult],
    overrides: dict[str, torch.Tensor] | None = None,
    label: str = "",
) -> None:
    """Writes config.json + model.safetensors, then the manifest. Quantised weights missing from results get RTN."""
    t0 = time.time()
    overrides = overrides or {}
    tensors: dict[str, torch.Tensor] = {}
    for name, shape in ckpt.shapes.items():
        if is_quantised_weight(name, shape):
            res = results.get(name)
            if res is None:
                log.warning("no quantisation result for %s: RTN", name)
                res = rtn(ckpt.tensor(name), qspec, device="cuda" if torch.cuda.is_available() else "cpu")
            prefix = name[: -len(".weight")]
            packed = packed_tensors(res.codes, res.scale, None if qspec.symmetric else res.zero, qspec.bits)
            for suffix, t in packed.items():
                tensors[f"{prefix}.{suffix}"] = t.contiguous()
        else:
            t = overrides.get(name)
            if t is None:
                t = ckpt.tensor(name)
            tensors[name] = t.to(torch.bfloat16).contiguous()
    config = dict(base_config)
    config["quantization_config"] = quantization_config(spec)
    common.remove_manifest(out)
    tmp = out / (WEIGHTS_FILE + ".tmp")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, out / WEIGHTS_FILE)
    common.atomic_write_bytes(out / "config.json", json.dumps(config, indent=2).encode())
    for stale in out.iterdir():
        if stale.name not in ("config.json", WEIGHTS_FILE) and stale.is_file():
            stale.unlink()
    common.write_manifest(out, task_id, "fixed", ["config.json", WEIGHTS_FILE])
    stage_dir = os.environ.get("QUANT_STAGE_DIR")  # local testing: keep a copy of every stage
    if stage_dir:
        import shutil

        dest = Path(stage_dir) / label
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", WEIGHTS_FILE):
            shutil.copy2(out / name, dest / name)
    log.info("checkpoint %s written in %.0fs (%.1f MB)", label, time.time() - t0, (out / WEIGHTS_FILE).stat().st_size / 1e6)


def load_packed(path: str, qspec: QSpec) -> dict[str, QResult]:
    """Read back a pack-quantized checkpoint written by write_checkpoint (dev: resume from a stage)."""
    from quant.packing import unpack_last_dim

    results = {}
    with safe_open(str(Path(path) / WEIGHTS_FILE), framework="pt") as sf:
        keys = set(sf.keys())
        for k in keys:
            if not k.endswith(".weight_packed"):
                continue
            prefix = k[: -len(".weight_packed")]
            out_f, in_f = sf.get_tensor(prefix + ".weight_shape").tolist()
            codes = unpack_last_dim(sf.get_tensor(k), qspec.bits, in_f).to(torch.uint8)
            scale = sf.get_tensor(prefix + ".weight_scale")
            zp = sf.get_tensor(prefix + ".weight_zero_point")
            zero = unpack_last_dim(zp.t().contiguous(), qspec.bits, out_f).t().contiguous().to(torch.uint8)
            results[prefix + ".weight"] = QResult(codes, scale, zero)
    return results


# ------------------------------------------------------------------------------------------------ evaluation


@torch.no_grad()
def mean_loss(model, rows: list[Row], max_rows: int | None = None) -> float:
    """Mean over rows of each row's mean supervised-token loss (the quant_loss convention)."""
    device = next(model.parameters()).device
    losses = []
    for row in rows[:max_rows]:
        ids = torch.tensor(row.input_ids, device=device).view(1, -1)
        labels = torch.tensor(row.labels, device=device)
        logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
        target = labels[1:]
        mask = target != IGNORE_INDEX
        if not mask.any():
            continue
        lp = torch.log_softmax(logits[mask], dim=-1)
        losses.append(float(-lp.gather(1, target[mask].view(-1, 1)).mean()))
    return sum(losses) / max(1, len(losses))


def ratio_proxy(model, rows: list[Row], max_rows: int) -> tuple[float, float]:
    """(our loss, loss / reference) on calibration rows that carry a reference loss."""
    rows = [r for r in rows if r.reference_loss is not None][:max_rows]
    if not rows:
        return float("nan"), float("nan")
    ours = mean_loss(model, rows)
    ref = sum(r.reference_loss for r in rows) / len(rows)
    return ours, ours / ref


def set_weights(model, results: dict[str, QResult], group: int) -> None:
    params = dict(model.named_parameters())
    for name, res in results.items():
        p = params.get(name)
        if p is not None:
            p.data.copy_(res.dequant(group, p.dtype).to(p.device))


# ------------------------------------------------------------------------------------------------ main


def _bap_budget(left: float, tune_reserve: float, size_b: float) -> float:
    """Block-AP's share of the time left; when tuning could not get a useful slot afterwards anyway (short budgets on
    large models: 7B at 0.5 h left ~13 min idle), Block-AP gets all of it but a write/eval margin."""
    budget = float(os.environ.get("QUANT_BAP_SHARE", 0.45)) * left
    if left - budget < tune_reserve + 600:
        budget = max(budget, left - 180 - 30 * size_b)
        log.info("Block-AP gets %.0fs of %.0fs left (no room for tuning after it)", budget, left)
    return budget


def run(args) -> None:
    deadline = common.Deadline.from_hours(args.hours_to_complete)
    spec = parse_spec(args)
    qspec = QSpec(bits=int(spec["weight_bits"]), group=int(spec["group_size"]), symmetric=bool(spec["symmetric"]))
    if qspec.symmetric:
        raise NotImplementedError("symmetric formats are not launch formats")
    out = common.output_dir(args.task_id, args.expected_repo_name)
    src = common.model_dir(args.model)
    log.info("fixed task %s: model %s spec %s, budget %.2fh (%.0fs usable)", args.task_id, src, spec, args.hours_to_complete, deadline.left())
    base_config = common.read_json(os.path.join(src, "config.json"))
    ckpt = InputCheckpoint(src)
    names = ckpt.quantised_names(qspec.group)
    bad = [n for n in names if ckpt.shapes[n][1] % qspec.group]
    if bad:
        log.warning("in_features not divisible by group size: %s", bad[:5])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. RTN fallback, straight from the files.
    t0 = time.time()
    rtn_results = {n: rtn(ckpt.tensor(n), qspec, device=device) for n in names}
    write_checkpoint(out, args.task_id, ckpt, base_config, qspec, spec, rtn_results, label="rtn")
    log.info("RTN done in %.0fs", time.time() - t0)

    # 2. data
    tok = common.load_tokenizer(src)
    calib, calib_meta = load_calibration(os.environ.get("QUANT_CALIBRATION_PATH"))
    dataset_type = json.loads(args.dataset_type or "{}")
    max_len = int(os.environ.get("QUANT_MAX_LEN", 2048))
    n_train = int(os.environ.get("QUANT_TRAIN_ROWS", 30000))
    train = load_train_rows(args.task_id, args.dataset, dataset_type, tok, max_len, n_train)

    model = common.load_model(src, device_map=device)
    n_eval = int(os.environ.get("QUANT_EVAL_ROWS", 300))
    eval_rows = [r for r in calib if r.reference_loss is not None]
    if eval_rows:
        bf16_loss, bf16_r = ratio_proxy(model, eval_rows, n_eval)
        log.info("bf16 on calibration: loss %.5f r %.4f (published ref %.5f)", bf16_loss, bf16_r, calib_meta.get("reference_loss", float("nan")))
    elif calib:
        eval_rows = calib
    else:
        eval_rows = train[-n_eval:]
        train = train[:-n_eval] if len(train) > 2 * n_eval else train
    score = lambda: ratio_proxy(model, eval_rows, n_eval) if eval_rows and eval_rows[0].reference_loss is not None else (mean_loss(model, eval_rows, n_eval), float("nan"))

    init_from = os.environ.get("QUANT_INIT_FROM")
    if init_from:
        rtn_results = load_packed(init_from, qspec)
        log.info("initialised from %s (%d linears)", init_from, len(rtn_results))
    set_weights(model, rtn_results, qspec.group)
    best_loss, best_r = score()
    best_name, best_results = "rtn", rtn_results
    log.info("RTN calibration loss %.5f r %.4f", best_loss, best_r)

    # 3. GPTQ
    gptq_rows = int(os.environ.get("QUANT_GPTQ_ROWS", 512))
    # Hessians from train rows; calibration rows stay a clean validation set unless train is too small.
    pool = train if len(train) >= 128 else calib + train
    samples = [torch.tensor(r.input_ids) for r in pool[:gptq_rows]]
    if samples and deadline.left() > 120 and os.environ.get("QUANT_GPTQ", "1") == "1":
        # restore bf16 weights before GPTQ
        params = dict(model.named_parameters())
        for n in names:
            p = params.get(n)
            if p is not None:
                p.data.copy_(ckpt.tensor(n).to(p.dtype).to(p.device))
        damp = float(os.environ.get("QUANT_DAMP", 0.01))
        # GPTQ only seeds Block-AP's scales/zeros (RTN seeds it nearly as well), so it gets a small share
        gptq_end = time.time() + float(os.environ.get("QUANT_GPTQ_SHARE", 0.12)) * deadline.left()
        t0 = time.time()
        gptq_results = quantize_model_gptq(
            model, samples, qspec, damp=damp, true_sequential=True, time_left=lambda: gptq_end - time.time()
        )
        log.info("GPTQ done in %.0fs", time.time() - t0)
        missing = [n for n in names if n not in gptq_results]
        for n in missing:
            gptq_results[n] = rtn_results[n]
        set_weights(model, gptq_results, qspec.group)
        loss, r = score()
        log.info("GPTQ calibration loss %.5f r %.4f", loss, r)
        if loss < best_loss:
            best_loss, best_r, best_name, best_results = loss, r, "gptq", gptq_results
            write_checkpoint(out, args.task_id, ckpt, base_config, qspec, spec, gptq_results, label="gptq")

    # 4. block-wise reconstruction (Block-AP), initialised from the best scales/zeros so far
    # final write + eval measured on H100: ~35 s for 7B (17 s write, 17 s eval); keep a wide margin
    size_b = sum(math.prod(ckpt.shapes[n]) for n in names) / 1e9
    tune_reserve = float(os.environ.get("QUANT_TUNE_RESERVE", 150 + 30 * size_b))
    if os.environ.get("QUANT_BLOCKAP", "1") == "1" and pool and deadline.left() > 900:
        from quant.blockap import blockap

        params = dict(model.named_parameters())
        for n in names:
            if n in params:
                params[n].data.copy_(ckpt.tensor(n).to(params[n].dtype).to(params[n].device))
        t0 = time.time()
        bap = blockap(
            model,
            pool,
            qspec,
            best_results,
            n_chunks=int(os.environ.get("QUANT_BAP_CHUNKS", 2048)),
            length=int(os.environ.get("QUANT_BAP_LEN", 512)),
            batch=int(os.environ.get("QUANT_BAP_BATCH", 4)),
            epochs=int(os.environ.get("QUANT_BAP_EPOCHS", 4)),
            lr_w=float(os.environ.get("QUANT_BAP_LR_W", 1e-4)),
            lr_q=float(os.environ.get("QUANT_BAP_LR_Q", 3e-4)),
            time_budget=_bap_budget(deadline.left(), tune_reserve, size_b),
        )
        if bap is not None:
            for n in names:
                bap.setdefault(n, best_results[n])
            set_weights(model, bap, qspec.group)
            loss, r = score()
            log.info("Block-AP calibration loss %.5f r %.4f (%.0fs)", loss, r, time.time() - t0)
            if loss < best_loss:
                best_loss, best_r, best_name, best_results = loss, r, "blockap", bap
                write_checkpoint(out, args.task_id, ckpt, base_config, qspec, spec, bap, label="blockap")
        set_weights(model, best_results, qspec.group)

    # 5. end-to-end scale tuning on train rows, codes frozen
    if os.environ.get("QUANT_TUNE", "1") == "1" and train and deadline.left() > tune_reserve + 300:
        from quant import tune

        set_weights(model, best_results, qspec.group)
        qlinears = tune.install(model, best_results, qspec.group)
        train_norms = os.environ.get("QUANT_TUNE_NORMS", "0") == "1"
        last_write = [time.time()]

        def evaluate():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return mean_loss(model, eval_rows, n_eval)

        def current():
            res = {k: q.result() for k, q in qlinears.items()}
            overrides = {}
            if train_norms:
                params = dict(model.named_parameters())
                overrides = {n: params[n].detach().to(torch.bfloat16).cpu() for n in ckpt.shapes if n in params and params[n].dim() == 1}
            return res, overrides

        def on_improve(value):
            # throttle mid-run writes; the final best is always written below
            if time.time() - last_write[0] > float(os.environ.get("QUANT_WRITE_EVERY", 900)) and deadline.left() > tune_reserve:
                res, ov = current()
                write_checkpoint(out, args.task_id, ckpt, base_config, qspec, spec, res, ov, label="tuned-partial")
                last_write[0] = time.time()

        teacher = None
        kd_alpha = float(os.environ.get("QUANT_KD", 0.0))
        if kd_alpha > 0:
            teacher = common.load_model(src, device_map=device)
            for p in teacher.parameters():
                p.requires_grad_(False)
        budget = deadline.left() - tune_reserve
        best_tuned = tune.train_scales(
            model,
            qlinears,
            train,
            evaluate,
            deadline.left,
            pad_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
            lr=float(os.environ.get("QUANT_TUNE_LR", 1e-3)),
            train_norms=train_norms,
            max_tokens=int(os.environ.get("QUANT_TUNE_TOKENS", 16384)),
            epochs=float(os.environ.get("QUANT_TUNE_EPOCHS", 3.0)),
            budget_seconds=budget,
            eval_every=int(os.environ.get("QUANT_EVAL_EVERY", 100)),
            on_improve=on_improve,
            teacher=teacher,
            kd_alpha=kd_alpha,
        )
        del teacher
        if best_tuned < best_loss:
            res, ov = current()
            ref = [r.reference_loss for r in eval_rows[:n_eval] if r.reference_loss is not None]
            best_r = best_tuned / (sum(ref) / len(ref)) if ref and len(ref) == len(eval_rows[:n_eval]) else float("nan")
            best_loss, best_name = best_tuned, "tuned"
            write_checkpoint(out, args.task_id, ckpt, base_config, qspec, spec, res, ov, label="tuned")

    log.info("final: %s calibration loss %.5f r %.4f, %.0fs elapsed", best_name, best_loss, best_r, deadline.elapsed())
    del model
    gc.collect()
