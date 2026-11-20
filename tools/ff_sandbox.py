"""Local freeform round trip through the validator's own code (G.O.D quant-tournament checkout), no network, no uploads.

  PYTHONPATH=/root/56/godq /mnt/sn56/quant/godvenv/bin/python tools/ff_sandbox.py \
      --repo /root/56/ours-quant --commit HEAD --output /mnt/sn56/quant/out/<task>/<repo> \
      --base /mnt/sn56/cache/models/<model> --task-id <task> --work /mnt/sn56/quant/ffsb/<name> [--cpuset 100-115]

Steps, each with the validator / trainer function that does it in production:
  1. enrolment: decoder_snapshot.archive_decoder_dir + unsafe_decoder_members + xz_bytes_of + check_decoder_runnable
  2. upload check on a copy of the output dir: trainer.quant_upload_check.check_quant_output (manifest, sizes, cap, prune)
  3. decode: job._extract_decoder of the .tar.xz, job.check_artifact_files, then sandbox.build_phase_spec /
     run_phase_spec run with sandbox.run_phase (same images, mounts, memory, CPUs, pids, read-only root, 900 s cap)
  4. output_check.check_decoded_output against the base model
  5. counted bytes: job.artifact_size_bytes + HF .gitattributes + decoder xz bytes; eval dir = base metadata
     (quant_artifacts.BASE_WEIGHT_PATTERNS excluded) + decoded shards, ready for tools/local_eval.py one (no spec).
Prints a JSON summary.
"""

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from fnmatch import fnmatch
from pathlib import Path

import docker

from core.models.task_models import TaskType
from trainer.quant_decode import job
from trainer.quant_decode import sandbox
from trainer.quant_decode.output_check import ExpectedTensors
from trainer.quant_decode.output_check import check_decoded_output
from trainer.quant_upload_check import check_quant_output
from trainer.quant_upload_check import quant_output_max_bytes
from validator.evaluation.quant_artifacts import BASE_WEIGHT_PATTERNS
from validator.tournament import decoder_snapshot as ds


HF_GITATTRIBUTES_BYTES = 1519


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cpuset", default="100-115")
    a = ap.parse_args()
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    summary = {}
    commit = subprocess.run(["git", "-C", a.repo, "rev-parse", a.commit], capture_output=True, text=True, check=True).stdout.strip()
    summary["commit"] = commit

    # 1. enrolment
    tar, xz = work / "decoder.tar", work / "decoder.tar.xz"
    ds.archive_decoder_dir(Path(a.repo), commit, tar)
    unsafe = ds.unsafe_decoder_members(tar)
    xz_bytes, _ = ds.xz_bytes_of(tar, xz)
    ds.check_decoder_runnable(tar)
    summary["enrolment"] = {"unsafe_members": unsafe, "decoder_xz_bytes": xz_bytes, "under_cap": xz_bytes <= ds.DECODER_MAX_XZ_BYTES}

    # 2. upload check on a copy
    art = work / "artifact"
    shutil.copytree(a.output, art)
    base = Path(a.base)
    verdict = check_quant_output(art, task_id=a.task_id, task_type=TaskType.QUANTFREEFORMTASK, max_bytes=quant_output_max_bytes(base))
    summary["upload_check"] = verdict.model_dump()
    if not verdict.ok:
        print(json.dumps(summary, indent=1))
        return
    (art / ".gitattributes").write_bytes(b"x" * HF_GITATTRIBUTES_BYTES)  # stand-in for the file HF adds (counted)

    # 3. decode in the sandbox
    dec, build, out = work / "decoder", work / "build", work / "output"
    for d in (dec, build, out):
        d.mkdir()
    job._extract_decoder(xz, dec)
    summary["artifact_files_check"] = job.check_artifact_files(art) or "ok"
    expected = ExpectedTensors.from_safetensors_dir(base)
    bf16 = job._bf16_bytes(expected)
    client = docker.from_env()
    phases = {}
    for spec in (
        sandbox.build_phase_spec(dec, build),
        sandbox.run_phase_spec(art, dec, out, bf16, build_dir=build),
    ):
        spec = spec.model_copy(update={"cpuset_cpus": a.cpuset})
        t0 = time.time()
        outcome = asyncio.run(sandbox.run_phase(client, spec))
        phases[spec.phase.value] = {"ok": outcome.ok, "failure": outcome.failure, "seconds": round(time.time() - t0, 1), "mem_limit": spec.mem_limit}
        if not outcome.ok:
            break
    summary["phases"] = phases
    if not all(p["ok"] for p in phases.values()):
        print(json.dumps(summary, indent=1))
        return

    # 4. output check
    check = check_decoded_output(out, expected)
    summary["output_check"] = check.model_dump()

    # 5. bytes and eval dir
    artifact_bytes = job.artifact_size_bytes(art)
    summary["artifact_bytes"] = artifact_bytes
    summary["total_bytes"] = artifact_bytes + xz_bytes
    summary["bf16_bytes"] = bf16
    summary["size_ratio"] = summary["total_bytes"] / bf16
    evald = work / "eval_model"
    evald.mkdir()
    for p in base.iterdir():
        if p.is_file() and not any(fnmatch(p.name, pat) for pat in BASE_WEIGHT_PATTERNS):
            shutil.copy2(p, evald / p.name)
    for p in out.glob("*.safetensors"):
        shutil.move(str(p), evald / p.name)
    summary["eval_dir"] = str(evald)
    print(json.dumps(summary, indent=1))
    (work / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
