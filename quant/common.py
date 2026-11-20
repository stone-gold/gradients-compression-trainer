"""Shared plumbing for both compression tracks: arguments, deadline, input model, calibration, manifest."""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger("quant")
START = time.time()
OUTPUT_ROOT = "/app/checkpoints"
CACHE_MODELS = "/cache/models"
MANIFEST = "quant_manifest.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="")
    p.add_argument("--dataset-type", default="{}")
    p.add_argument("--task-type", required=True)
    p.add_argument("--file-format", default="s3")
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, default=1.0)
    p.add_argument("--quant-spec", default=None)
    # Anything new the validator adds must not crash us.
    args, unknown = p.parse_known_args(argv)
    if unknown:
        log.warning("ignoring unknown arguments: %s", unknown)
    return args


@dataclass
class Deadline:
    """Wall-clock budget. end is the container kill time minus a safety margin."""

    end: float

    @classmethod
    def from_hours(cls, hours: float) -> "Deadline":
        total = hours * 3600.0
        margin = float(os.environ.get("QUANT_SAFETY_SECONDS", max(300.0, 0.06 * total)))
        return cls(end=START + total - margin)

    def left(self) -> float:
        return self.end - time.time()

    def elapsed(self) -> float:
        return time.time() - START


def model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    path = os.path.join(CACHE_MODELS, model.replace("/", "--"))
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"input model not found at {path}")


def output_dir(task_id: str, repo_name: str) -> Path:
    root = os.environ.get("QUANT_OUTPUT_ROOT", OUTPUT_ROOT)
    path = Path(root) / task_id / repo_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def quant_target() -> dict:
    raw = os.environ.get("QUANT_TARGET")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_manifest(out: Path, task_id: str, track: str, files: list[str]) -> None:
    """Written last: lists exactly `files` with their current sizes. No exec bits, no symlinks."""
    entries = []
    for name in files:
        path = out / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"manifest file {name} is not a regular file")
        os.chmod(path, 0o644)
        entries.append({"name": name, "bytes": path.stat().st_size})
    doc = {"version": 1, "task_id": task_id, "track": track, "files": entries}
    atomic_write_bytes(out / MANIFEST, json.dumps(doc, indent=1).encode())
    os.chmod(out / MANIFEST, 0o644)
    log.info("manifest written: %s", [(e["name"], e["bytes"]) for e in entries])


def remove_manifest(out: Path) -> None:
    try:
        (out / MANIFEST).unlink()
    except FileNotFoundError:
        pass


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=False)


def load_model(path: str, device_map="cpu"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device_map, trust_remote_code=False, low_cpu_mem_usage=True
    )
    model.eval()
    model.config.use_cache = False
    return model
