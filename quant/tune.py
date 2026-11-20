"""End-to-end tuning of the quantised model with integer codes frozen (EfficientQAT's E2E-QP, scales only).

Every quantised Linear becomes a QLinear: w = (code - zero) * (scale0 * mult), codes and zero points fixed, mult a
trainable fp32 multiplier per group (init 1). Optionally the bf16 norm weights and biases train too. The loss is the
evaluator's: next-token CE on supervised tokens of the task's train rows (or KL to the bf16 teacher).
"""

import logging
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from quant.data import IGNORE_INDEX
from quant.data import Row
from quant.quantize import QResult


log = logging.getLogger("quant.tune")


class _QMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qz, scale, group):
        out_f, in_f = qz.shape
        w = (qz.view(out_f, -1, group).to(x.dtype) * scale.to(x.dtype).unsqueeze(-1)).view(out_f, in_f)
        ctx.save_for_backward(x, qz, scale)
        ctx.group = group
        return x.matmul(w.t())

    @staticmethod
    def backward(ctx, g):
        x, qz, scale = ctx.saved_tensors
        group = ctx.group
        out_f, in_f = qz.shape
        w = (qz.view(out_f, -1, group).to(x.dtype) * scale.to(x.dtype).unsqueeze(-1)).view(out_f, in_f)
        grad_x = g.matmul(w)
        g2 = g.reshape(-1, out_f)
        x2 = x.reshape(-1, in_f)
        grad_w = g2.t().float().matmul(x2.float())  # [out, in]
        grad_scale = (grad_w.view(out_f, -1, group) * qz.view(out_f, -1, group).float()).sum(-1)
        return grad_x, None, grad_scale, None


class QLinear(nn.Module):
    def __init__(self, linear: nn.Linear, res: QResult, group: int):
        super().__init__()
        dev = linear.weight.device
        codes = res.codes.to(dev).to(torch.int16)
        zero = res.zero.to(dev).to(torch.int16).repeat_interleave(group, dim=1)
        self.register_buffer("qz", (codes - zero).to(torch.int8), persistent=False)
        self.register_buffer("scale0", res.scale.to(dev).float(), persistent=False)
        self.mult = nn.Parameter(torch.ones_like(self.scale0))
        self.bias = linear.bias
        self.group = group
        self.res = res

    def scale(self) -> torch.Tensor:
        # what the checkpoint stores: bf16
        return (self.scale0 * self.mult).to(torch.bfloat16)

    def forward(self, x):
        s = self.scale0 * self.mult
        s = s + (s.to(torch.bfloat16).float() - s).detach()  # bf16 rounding, straight-through
        y = _QMatmul.apply(x, self.qz, s, self.group)
        if self.bias is not None:
            y = y + self.bias
        return y

    def result(self) -> QResult:
        return QResult(self.res.codes, self.scale().detach().cpu(), self.res.zero)


def install(model, results: dict[str, QResult], group: int) -> dict[str, QLinear]:
    """Replace every quantised nn.Linear (by checkpoint weight name) with a QLinear. Returns name -> QLinear."""
    qlinears = {}
    for name, module in list(model.named_modules()):
        key = name + ".weight"
        if isinstance(module, nn.Linear) and key in results:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            q = QLinear(module, results[key], group)
            setattr(parent, child, q)
            qlinears[key] = q
            module.weight = None
    torch.cuda.empty_cache()
    return qlinears


def _batches(rows: list[Row], max_tokens: int, rng: random.Random):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].input_ids))
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(rows[i].input_ids)
        if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def _collate(rows: list[Row], idx: list[int], pad_id: int, device):
    n = max(len(rows[i].input_ids) for i in idx)
    ids = torch.full((len(idx), n), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), n), IGNORE_INDEX, dtype=torch.long)
    att = torch.zeros((len(idx), n), dtype=torch.long)
    for b, i in enumerate(idx):
        r = rows[i]
        L = len(r.input_ids)
        ids[b, :L] = torch.tensor(r.input_ids)
        labels[b, :L] = torch.tensor(r.labels)
        att[b, :L] = 1
    return ids.to(device), labels.to(device), att.to(device)


def _ce_chunk(h, w, t):
    return F.cross_entropy(F.linear(h, w).float(), t, reduction="none")


def supervised_ce(model, ids, labels, att, chunk: int = 4096):
    """Mean over rows of each row's mean supervised-token CE, computing logits only at supervised positions, chunked
    and recomputed in backward so the [tokens, vocab] logits never sit in memory."""
    decoder = model.get_decoder()
    head = model.get_output_embeddings()
    hs = decoder(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    tg = labels[:, 1:]
    mask = tg != IGNORE_INDEX
    n_row = mask.sum(1)
    rows = n_row > 0
    weight = (mask / n_row.clamp(min=1).unsqueeze(1))[mask]
    h = hs[:, :-1][mask]
    t = tg[mask]
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        ce = torch.utils.checkpoint.checkpoint(_ce_chunk, h[i : i + chunk], head.weight, t[i : i + chunk], use_reentrant=False)
        total = total + (ce * weight[i : i + chunk]).sum()
    return total / rows.sum().clamp(min=1)


def per_row_ce(logits, labels):
    """[B] mean supervised-token CE per row (shifted), and a mask of rows with a supervised token."""
    lg = logits[:, :-1].float()
    tg = labels[:, 1:]
    mask = tg != IGNORE_INDEX
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.clamp(min=0).reshape(-1), reduction="none").view(tg.shape)
    ce = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
    return ce, mask.any(1)


def train_scales(
    model,
    qlinears: dict[str, QLinear],
    train_rows: list[Row],
    evaluate,
    deadline_left,
    pad_id: int,
    lr: float = 5e-4,
    train_norms: bool = False,
    max_tokens: int = 16384,
    epochs: float = 1.0,
    budget_seconds: float = float("inf"),
    eval_every: int = 100,
    on_improve=None,
    seed: int = 0,
):
    """Adam on the scale multipliers (and optionally norms/biases); keeps the best state by evaluate() (lower=better)."""
    device = next(p for p in model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    params = [q.mult for q in qlinears.values()]
    for q in params:
        q.requires_grad_(True)
    extra = []
    if train_norms:
        for name, p in model.named_parameters():
            if ("norm" in name or name.endswith(".bias")) and p.dim() == 1:
                p.data = p.data.float() if p.dtype != torch.float32 else p.data
                extra.append(p)
                p.requires_grad_(True)
    groups = [{"params": params, "lr": lr}]
    if extra:
        groups.append({"params": extra, "lr": lr * 0.2})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    rng = random.Random(seed)
    batches = _batches(train_rows, max_tokens, rng)
    total = max(1, int(len(batches) * epochs))
    warm = max(1, total // 20)
    best = evaluate()
    log.info("tune start: %d batches x %.2f epochs, eval %.5f", len(batches), epochs, best)
    snapshot = lambda: ([q.mult.detach().clone() for q in qlinears.values()], [p.detach().clone() for p in extra])
    best_state = snapshot()
    t0 = time.time()
    step = 0
    step_time = None
    while step < total:
        idx = batches[step % len(batches)]
        ts = time.time()
        frac = step / total
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = (lr if gi == 0 else lr * 0.2) * f
        ids, labels, att = _collate(train_rows, idx, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = supervised_ce(model, ids, labels, att)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        dt = time.time() - ts
        step_time = dt if step_time is None or step <= 3 else 0.9 * step_time + 0.1 * dt
        if step == 8:
            fit = int(budget_seconds / (step_time * 1.15))
            if fit < total:
                log.info("time budget %.0fs fits %d of %d steps", budget_seconds, fit, total)
                total = max(step + 1, fit)
                warm = min(warm, max(1, total // 20))
        if step % 20 == 0:
            log.info("step %d/%d loss %.4f lr %.2e %.2fs/step", step, total, float(loss), opt.param_groups[0]["lr"], step_time)
        last = step >= total or deadline_left() < step_time * eval_every + 120
        if step % eval_every == 0 or last:
            model.eval()
            cur = evaluate()
            model.train()
            log.info("step %d eval %.5f (best %.5f) %.0fs", step, cur, best, time.time() - t0)
            if cur < best:
                best = cur
                best_state = snapshot()
                if on_improve is not None:
                    on_improve(best)
            if last:
                break
    # restore best
    for q, m in zip(qlinears.values(), best_state[0]):
        q.mult.data.copy_(m)
    for p, v in zip(extra, best_state[1]):
        p.data.copy_(v)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return best
