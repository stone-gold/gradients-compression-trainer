"""Local replica of the fixed-track quant evaluator (run inside an image built from G.O.D's quant-evaluator.dockerfile).

  python local_eval.py ref  --base B --test T --calib-in C.jsonl --calib-out C.json --task-id ID
      bf16 reference loss on the test set, and per-row reference losses on the calibration rows -> QuantCalibrationSet.
  python local_eval.py eval --base B --test T --spec JSON --models M1 M2 ... [--ref-loss X]
      per model: structural check (expected_fixed_tensor_set), config checks (quant_config_failure +
      config_drift_failure on vLLM-resolved configs), vLLM load + kernel check, test loss -> r.
Each model is scored in its own subprocess (one vLLM engine per process), as the official evaluator does.
"""

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path


def headers(path: Path) -> dict:
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                tensors[k] = {"dtype": v["dtype"], "shape": v["shape"]}
    return tensors


def score_one(model: str, base: str, test: str, spec_json: str | None) -> dict:
    from core.models.quant_models import QuantFixedSpec
    from validator.evaluation.evaluators.quant import FixedConfigCheck
    from validator.evaluation.evaluators.quant import evaluate_repo
    from validator.evaluation.quant_loss import load_examples
    from validator.evaluation.quant_structure import TensorSet
    from validator.evaluation.quant_structure import TensorSpec
    from validator.evaluation.quant_structure import check_fixed_submission
    from validator.evaluation.quant_structure import expected_fixed_tensor_set
    from validator.evaluation.quant_vllm import VllmPromptLogprobEngine
    from validator.evaluation.quant_vllm import checked_kernel_classes
    from validator.evaluation.quant_vllm import disable_humming_stream_k
    from validator.evaluation.quant_vllm import expected_kernels_for
    from validator.evaluation.quant_vllm import resolved_model_config

    disable_humming_stream_k()
    import os

    util = os.environ.get("LOCAL_VLLM_GPU_UTIL")  # local only: share a GPU with another process (does not change logprobs)
    if util:
        import functools

        import validator.evaluation.quant_vllm as qv

        qv.LLM = functools.partial(qv.LLM, gpu_memory_utilization=float(util))
        VllmPromptLogprobEngine = qv.VllmPromptLogprobEngine
    spec = QuantFixedSpec.model_validate_json(spec_json) if spec_json else None
    result = {"model": model}
    if spec is not None:
        to_set = lambda h: TensorSet(tensors={k: TensorSpec(**v) for k, v in h.items()})
        base_config = json.loads((Path(base) / "config.json").read_text())
        expected = expected_fixed_tensor_set(to_set(headers(Path(base))), base_config, spec)
        result["structural"] = check_fixed_submission(to_set(headers(Path(model))), expected) or "ok"
    examples = load_examples(Path(test))
    max_len = max(len(e.input_ids) for e in examples)
    expected_kernels = expected_kernels_for(spec) if spec is not None else None
    t0 = time.time()
    res = evaluate_repo(
        repo=model,
        examples=examples,
        engine_factory=lambda path: VllmPromptLogprobEngine(path, max_len, base),
        layer_check=lambda engine: checked_kernel_classes(engine.llm, spec, expected_kernels),
        quant_spec=spec,
        config_loader=lambda repo: json.loads((Path(repo) / "config.json").read_text()),
        config_check=FixedConfigCheck(base_model=base, resolve=resolved_model_config),
    )
    result["seconds"] = round(time.time() - t0, 1)
    result["eval_loss"] = res.get("eval_loss")
    result["quant_failure"] = res.get("quant_failure")
    result["per_example_losses"] = res.get("per_example_losses")
    kc = res.get("kernel_classes") or {}
    result["kernels"] = sorted(set(kc.values()))
    result["n_quant_layers"] = len(kc)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "eval", "one"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--spec")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--ref-loss", type=float)
    ap.add_argument("--calib-in")
    ap.add_argument("--calib-out")
    ap.add_argument("--task-id", default="local")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.mode == "one":
        res = score_one(a.models[0], a.base, a.test, a.spec)
        Path(a.out).write_text(json.dumps(res))
        return

    if a.mode == "ref":
        tmp = Path(a.calib_out).with_suffix(".tmp.json")
        out = {}
        for name, test in (("test", a.test), ("calib", a.calib_in)):
            subprocess.run([sys.executable, __file__, "one", "--base", a.base, "--test", test, "--models", a.base, "--out", str(tmp)], check=True)
            out[name] = json.loads(tmp.read_text())
        rows = [json.loads(l) for l in Path(a.calib_in).read_text().splitlines() if l.strip()]
        per = out["calib"]["per_example_losses"]
        examples = [
            {"input_ids": r["input_ids"], "labels": r["labels"], "reference_loss": None if l != l else l} for r, l in zip(rows, per)
        ]
        doc = {
            "task_id": a.task_id,
            "eval_set_fingerprint": "local",
            "reference_model": a.base,
            "reference_loss": out["calib"]["eval_loss"],
            "reference_sku": "local",
            "examples": examples,
        }
        Path(a.calib_out).write_text(json.dumps(doc))
        Path(a.calib_out).with_suffix(".ref.json").write_text(
            json.dumps({"test_loss": out["test"]["eval_loss"], "calib_loss": out["calib"]["eval_loss"], "test_per_example": out["test"]["per_example_losses"]})
        )
        print(json.dumps({"test_ref_loss": out["test"]["eval_loss"], "calib_ref_loss": out["calib"]["eval_loss"]}))
        return

    ref = a.ref_loss
    for m in a.models:
        tmp = Path("/tmp/one.json")
        p = subprocess.run([sys.executable, __file__, "one", "--base", a.base, "--test", a.test, "--spec", a.spec, "--models", m, "--out", str(tmp)])
        if p.returncode != 0:
            print(json.dumps({"model": m, "error": f"exit {p.returncode}"}))
            continue
        res = json.loads(tmp.read_text())
        res.pop("per_example_losses", None)
        if ref and res.get("eval_loss"):
            res["r"] = res["eval_loss"] / ref
        print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
