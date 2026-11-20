"""Build a local fixed-track compression task from a real text task (run inside the quant evaluator image, CPU).

Outputs in --out:
  train_data.json      raw train rows minus the calibration rows (the trainer's /cache/datasets/{task}_train_data.json)
  calib_examples.jsonl tokenised calibration rows (reference losses are added by local_eval.py ref)
  test.jsonl           tokenised held-out rows, as validator quant prep builds them
  dataset_type.json    the --dataset-type JSON
"""

import argparse
import json
import random
from pathlib import Path

from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from validator.tasks.quant_eval_set import build_quant_eval_set
from validator.tasks.quant_eval_set import quant_eval_max_sequence_len
from validator.tasks.quant_eval_set import quant_eval_set_jsonl
from validator.tasks.quant_tokenizer import load_quant_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-calib", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    task_dir, out = Path(a.task_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    task = json.loads((task_dir / "task.json").read_text())
    if task["task_type"] == "ChatTask":
        dt = ChatTemplateDatasetType(**{k: task.get(k) for k in ChatTemplateDatasetType.model_fields if task.get(k) is not None})
    else:
        dt = InstructTextDatasetType(**{k: task.get(k) for k in InstructTextDatasetType.model_fields if task.get(k) is not None})
    (out / "dataset_type.json").write_text(dt.model_dump_json())
    train = json.loads((task_dir / "training_data.json").read_text())
    test = json.loads((task_dir / "test_data.json").read_text())
    rng = random.Random(a.seed)
    rng.shuffle(train)
    rng.shuffle(test)
    tok = load_quant_tokenizer(a.model)
    max_len = quant_eval_max_sequence_len(tok.context_len, 4096)
    calib_rows, train_rows = train[: a.n_calib * 2], train[a.n_calib * 2 :]
    calib = build_quant_eval_set(calib_rows, dt, tok, max_len).examples[: a.n_calib]
    test_build = build_quant_eval_set(test[: a.n_test * 2], dt, tok, max_len)
    (out / "calib_examples.jsonl").write_text(quant_eval_set_jsonl(calib))
    (out / "test.jsonl").write_text(quant_eval_set_jsonl(test_build.examples[: a.n_test]))
    (out / "train_data.json").write_text(json.dumps(train_rows))
    print(f"{task['task_id']}: {len(train_rows)} train rows, {len(calib)} calib, {min(a.n_test, len(test_build.examples))} test")


if __name__ == "__main__":
    main()
