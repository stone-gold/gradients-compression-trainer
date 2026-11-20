"""Task data -> tokenised (input_ids, labels) rows, labelled the way the quant evaluator labels its test set.

Mirrors G.O.D validator/tasks/quant_eval_set.py (instruct rows: axolotl user_defined prompt + completion tokenised
apart; chat rows: chat_labels.label_chat, vendored verbatim in quant/vendor). Apache-2.0, see NOTICE.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

from quant.vendor.chat_labels import ChatLabelSkip
from quant.vendor.chat_labels import chat_messages
from quant.vendor.chat_labels import chat_role_map
from quant.vendor.chat_labels import label_chat


log = logging.getLogger("quant.data")
IGNORE_INDEX = -100
CHATML_PATH = Path(__file__).parent / "vendor" / "chatml.jinja"


@dataclass
class Row:
    input_ids: list[int]
    labels: list[int]
    reference_loss: float | None = None

    @property
    def n_scored(self) -> int:
        return sum(1 for label in self.labels[1:] if label != IGNORE_INDEX)


def _text(value) -> str:
    return "" if value is None else str(value)


def instruct_pair(row: dict, dt: dict) -> tuple[str, str] | None:
    f_ins = dt.get("field_instruction") or "instruction"
    f_inp = dt.get("field_input") or "input"
    f_out = dt.get("field_output") or "output"
    f_sys = dt.get("field_system") or "system"
    completion = _text(row.get(f_out))
    if not completion:
        return None
    instruction = _text(row.get(f_ins))
    input_text = _text(row.get(f_inp))
    if input_text:
        turn_format = dt.get("format") or ("{instruction} {input}" if dt.get("field_input") else "{instruction}")
    else:
        turn_format = dt.get("no_input_format") or "{instruction}"
    system = _text(row[f_sys]) if f_sys in row else _text(dt.get("system_prompt"))
    system_format = dt.get("system_format")
    prefix = system_format.format(system=system) if system and system_format else ""
    return prefix + turn_format.format(instruction=instruction, input=input_text), completion


def tokenize_instruct(tok, prompt: str, completion: str) -> Row:
    prompt_ids = tok.encode(prompt, add_special_tokens=True) if prompt else []
    completion_ids = tok.encode(completion, add_special_tokens=True)
    if completion_ids and completion_ids[0] == tok.bos_token_id:
        completion_ids = completion_ids[1:]
    if tok.eos_token_id is not None and (not completion_ids or completion_ids[-1] != tok.eos_token_id):
        completion_ids.append(tok.eos_token_id)
    return Row(prompt_ids + completion_ids, [IGNORE_INDEX] * len(prompt_ids) + completion_ids)


class _HfChat:
    def __init__(self, tok, template: str):
        self.tok = tok
        self.template = template

    def render(self, messages, add_generation_prompt):
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, chat_template=self.template
        )

    def encode_with_offsets(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [(int(a), int(b)) for a, b in enc["offset_mapping"]]


def _resolve_template(name: str | None, tok) -> str:
    choice = name or "tokenizer_default"
    own = getattr(tok, "chat_template", None)
    if isinstance(own, dict):
        own = own.get("default")
    if choice == "tokenizer_default":
        if not own:
            raise ValueError("tokenizer has no chat template")
        return own
    if choice.startswith("tokenizer_default_fallback_"):
        if own:
            return own
        choice = choice[len("tokenizer_default_fallback_"):]
    if choice != "chatml":
        raise ValueError(f"unknown chat template {choice}")
    return CHATML_PATH.read_text(encoding="utf-8")


def is_chat(dt: dict) -> bool:
    return "chat_column" in dt or "chat_template" in dt


def tokenize_rows(rows: list[dict], dt: dict, tok, max_len: int) -> list[Row]:
    out = []
    if is_chat(dt):
        labeller = _HfChat(tok, _resolve_template(dt.get("chat_template"), tok))
        role_map = chat_role_map(dt.get("chat_user_reference"), dt.get("chat_assistant_reference"))
    for raw in rows:
        try:
            if is_chat(dt):
                messages = chat_messages(
                    raw,
                    dt.get("chat_column") or "conversations",
                    dt.get("chat_role_field") or "role",
                    dt.get("chat_content_field") or "content",
                    role_map,
                )
                labelled = label_chat(labeller, messages)
                if isinstance(labelled, ChatLabelSkip):
                    continue
                row = Row(labelled.input_ids, labelled.labels)
            else:
                pair = instruct_pair(raw, dt)
                if pair is None:
                    continue
                row = tokenize_instruct(tok, *pair)
        except Exception:  # noqa: BLE001 - a bad row is skipped, never fatal
            continue
        if row.n_scored == 0 or len(row.input_ids) > max_len:
            continue
        out.append(row)
    return out


def load_calibration(path: str | None) -> tuple[list[Row], dict]:
    if not path or not os.path.exists(path):
        log.warning("no calibration file at %r", path)
        return [], {}
    with open(path) as f:
        doc = json.load(f)
    rows = [Row(e["input_ids"], e["labels"], e.get("reference_loss")) for e in doc.get("examples", [])]
    meta = {k: v for k, v in doc.items() if k != "examples"}
    log.info("calibration: %d rows, reference_loss %s", len(rows), meta.get("reference_loss"))
    return rows, meta


def load_train_rows(task_id: str, dataset_arg: str, dt: dict, tok, max_len: int, limit: int, seed: int = 0) -> list[Row]:
    candidates = [f"/cache/datasets/{task_id}_train_data.json", dataset_arg]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        log.warning("no train split found (%s)", candidates)
        return []
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = next((v for v in rows.values() if isinstance(v, list)), [])
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = tokenize_rows(rows[: limit * 2], dt, tok, max_len)[:limit]
    log.info("train split %s: %d rows on disk, %d tokenised", path, len(rows), len(out))
    return out
