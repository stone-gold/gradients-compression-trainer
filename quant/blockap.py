"""Block-wise reconstruction training (EfficientQAT Block-AP style), layer by layer.

For each decoder layer: latent weights W, scales s and continuous zero points z of every Linear train with
straight-through rounding to minimise MSE between the quantised layer on quantised-prefix inputs and the bf16
layer on bf16 inputs. Final codes = clamp(round(W/s) + round(z)), scales rounded to bf16 (also during training).
"""

import logging
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.quantize import QResult
from quant.quantize import QSpec
from quant.quantize import decoder_layers
from quant.quantize import module_names


log = logging.getLogger("quant.blockap")


def _round_ste(x):
    return x + (torch.round(x) - x).detach()


def _bf16_ste(x):
    return x + (x.to(torch.bfloat16).float() - x).detach()


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, init: QResult, spec: QSpec):
        super().__init__()
        g = spec.group
        self.spec = spec
        self.weight = nn.Parameter(linear.weight.detach().float().clone())
        self.scale = nn.Parameter(init.scale.to(linear.weight.device).float().clone())
        self.zero = nn.Parameter(init.zero.to(linear.weight.device).float().clone())
        self.bias = linear.bias
        self.out_f, self.in_f = linear.weight.shape

    def qweight(self):
        g, maxq = self.spec.group, self.spec.maxq
        s = _bf16_ste(self.scale.clamp(min=1e-6))
        z = _round_ste(self.zero).clamp(0, maxq)
        w = self.weight.view(self.out_f, -1, g)
        q = torch.clamp(_round_ste(w / s.unsqueeze(-1)) + z.unsqueeze(-1), 0, maxq)
        return ((q - z.unsqueeze(-1)) * s.unsqueeze(-1)).view(self.out_f, self.in_f)

    def forward(self, x):
        return F.linear(x, self.qweight().to(x.dtype), self.bias)

    @torch.no_grad()
    def result(self) -> QResult:
        g, maxq = self.spec.group, self.spec.maxq
        s = self.scale.clamp(min=1e-6).to(torch.bfloat16).float()
        z = torch.round(self.zero).clamp(0, maxq)
        w = self.weight.view(self.out_f, -1, g)
        q = torch.clamp(torch.round(w / s.unsqueeze(-1)) + z.unsqueeze(-1), 0, maxq).view(self.out_f, self.in_f)
        return QResult(q.to(torch.uint8).cpu(), s.to(torch.bfloat16).cpu(), z.to(torch.uint8).cpu())


def make_chunks(rows, n_chunks: int, length: int, seed: int = 0) -> torch.Tensor:
    """Concatenate token streams of rows into n_chunks x length."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    stream: list[int] = []
    for i in order:
        stream.extend(rows[i].input_ids)
        if len(stream) >= n_chunks * length:
            break
    n = min(n_chunks, len(stream) // length)
    return torch.tensor(stream[: n * length]).view(n, length)


class _Stop(Exception):
    pass


@torch.no_grad()
def _capture(model, layers, chunks, batch, device):
    hiddens, kw = [], {}

    def hook(_m, args, kwargs):
        args = list(args)
        h = args.pop(0) if args else kwargs.pop("hidden_states")
        if not kw:
            kw.update({k: v for k, v in kwargs.items() if k not in ("past_key_value", "past_key_values", "use_cache")})
            kw["_args"] = tuple(args)
        hiddens.append(h)
        raise _Stop

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for i in range(0, len(chunks), batch):
            try:
                model(input_ids=chunks[i : i + batch].to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()
    args = kw.pop("_args")
    return hiddens, args, kw


def _out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def blockap(
    model,
    rows,
    spec: QSpec,
    init: dict[str, QResult],
    n_chunks: int = 256,
    length: int = 512,
    batch: int = 4,
    epochs: int = 2,
    lr_w: float = 2e-5,
    lr_q: float = 1e-4,
    time_budget: float = float("inf"),
) -> dict[str, QResult] | None:
    """Returns {weight name: QResult} or None when the time budget cannot fit. Model weights end dequantised."""
    device = next(model.parameters()).device
    layers = decoder_layers(model)
    names = module_names(model)
    # three activation streams (quantised inputs, bf16 inputs, bf16 targets) live on the GPU: cap them at ~30% of free memory
    hidden = model.config.get_text_config().hidden_size
    free, _ = torch.cuda.mem_get_info()
    cap = int(0.30 * free / (3 * length * hidden * 2))
    if cap < n_chunks:
        log.info("block-AP: %d chunks fit in memory (asked %d)", cap, n_chunks)
        n_chunks = max(batch, cap)
    chunks = make_chunks(rows, n_chunks, length)
    n_chunks = (len(chunks) // batch) * batch
    chunks = chunks[:n_chunks]
    if n_chunks == 0:
        return None
    t0 = time.time()
    x_q, args, kwargs = _capture(model, layers, chunks, batch, device)
    x_fp = [h.clone() for h in x_q]
    results: dict[str, QResult] = {}
    nb = len(x_q)
    for li, layer in enumerate(layers):
        tl = time.time()
        with torch.no_grad():
            y_fp = [_out(layer(x_fp[b], *args, **kwargs)) for b in range(nb)]
        linears = {
            names[id(m)]: m for m in layer.modules() if isinstance(m, nn.Linear) and (names[id(m)] + ".weight") in init
        }
        fq = {}
        for n, m in linears.items():
            parent_name, _, child = n.rpartition(".")
            parent = model.get_submodule(parent_name)
            f = FakeQuantLinear(m, init[n + ".weight"], spec)
            setattr(parent, child, f)
            fq[n] = (parent, child, m, f)
        for p in layer.parameters():
            p.requires_grad_(False)
        wparams = [f.weight for (_, _, _, f) in fq.values()]
        qparams = [p for (_, _, _, f) in fq.values() for p in (f.scale, f.zero)]
        for p in wparams + qparams:
            p.requires_grad_(True)
        opt = torch.optim.AdamW([{"params": wparams, "lr": lr_w}, {"params": qparams, "lr": lr_q}], weight_decay=0.0)
        steps = epochs * nb
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        layer.train()
        first = last = None
        for step in range(steps):
            b = step % nb
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = _out(layer(x_q[b], *args, **kwargs))
            loss = F.mse_loss(y.float(), y_fp[b].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step < nb:
                first = float(loss) if first is None else first + float(loss)
            if step >= steps - nb:
                last = float(loss) if last is None else last + float(loss)
        layer.eval()
        for n, (parent, child, m, f) in fq.items():
            res = f.result()
            results[n + ".weight"] = res
            m.weight.data.copy_(res.dequant(spec.group, m.weight.dtype).to(m.weight.device))
            setattr(parent, child, m)
        del fq, opt, wparams, qparams
        with torch.no_grad():
            x_q = [_out(layer(x_q[b], *args, **kwargs)) for b in range(nb)]
        x_fp = y_fp
        dt = time.time() - tl
        log.info("blockap layer %d/%d mse %.3e -> %.3e (%.1fs)", li + 1, len(layers), first / nb, last / nb, dt)
        if li == 0:
            per_epoch = dt / epochs
            fit = int((time_budget - dt) / max(1e-6, per_epoch * (len(layers) - 1)))
            if fit < 1:
                log.warning("block-AP needs ~%.0fs > budget %.0fs: abort", per_epoch * len(layers), time_budget)
                return None
            if fit < epochs:
                log.info("block-AP: %d epochs per layer fit the budget (asked %d)", fit, epochs)
                epochs = fit
    torch.cuda.empty_cache()
    log.info("block-AP done in %.0fs", time.time() - t0)
    return results
