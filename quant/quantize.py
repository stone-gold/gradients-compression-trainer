"""Asymmetric group-wise weight quantisation: RTN with clip search, and GPTQ (static groups, no act-order).

Every quantised Linear ends as (codes uint8 [out, in], scale bf16 [out, G], zero uint8 [out, G]) with
dequant w = (code - zero) * scale, exactly what the pack-quantized checkpoint stores.
"""

import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


log = logging.getLogger("quant.quantize")


@dataclass
class QSpec:
    bits: int
    group: int
    symmetric: bool = False

    @property
    def maxq(self) -> int:
        return (1 << self.bits) - 1


@dataclass
class QResult:
    codes: torch.Tensor  # uint8 [out, in]
    scale: torch.Tensor  # bf16 [out, G]
    zero: torch.Tensor  # uint8 [out, G]

    def dequant(self, group: int, dtype=torch.bfloat16) -> torch.Tensor:
        s = self.scale.float().repeat_interleave(group, dim=1)
        z = self.zero.float().repeat_interleave(group, dim=1)
        return ((self.codes.float() - z) * s).to(dtype)


def clip_grid(bits: int) -> list[float]:
    lo = 0.45 if bits <= 2 else 0.6
    return [1.0 - i * (1.0 - lo) / 24 for i in range(25)]


@torch.no_grad()
def group_params(w: torch.Tensor, spec: QSpec, weight: torch.Tensor | None = None, grid=None):
    """w [R, gs] fp32 -> (scale [R] fp32 (bf16-representable), zero [R] fp32 integer), best clip per row.

    weight [gs] (Hessian diagonal) weights the reconstruction error per input column.
    """
    maxq = spec.maxq
    xmin = w.min(dim=1).values
    xmax = w.max(dim=1).values
    grid = grid or clip_grid(spec.bits)
    best_err = torch.full_like(xmin, float("inf"))
    best_s = torch.zeros_like(xmin)
    best_z = torch.zeros_like(xmin)
    for p in grid:
        lo, hi = xmin * p, xmax * p
        s = ((hi - lo) / maxq).clamp(min=1e-10).to(torch.bfloat16).float()
        s = torch.where(s > 0, s, torch.full_like(s, 1e-8))
        z = torch.round(-lo / s).clamp(0, maxq)
        q = torch.clamp(torch.round(w / s[:, None]) + z[:, None], 0, maxq)
        err = ((q - z[:, None]) * s[:, None] - w).pow(2)
        if weight is not None:
            err = err * weight[None, :]
        err = err.sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
        best_z = torch.where(better, z, best_z)
    return best_s, best_z


@torch.no_grad()
def rtn(weight: torch.Tensor, spec: QSpec, hdiag: torch.Tensor | None = None, device="cuda") -> QResult:
    w = weight.to(device=device, dtype=torch.float32)
    out_f, in_f = w.shape
    g = spec.group
    groups = in_f // g
    wg = w.view(out_f * groups, g)
    hw = None
    if hdiag is not None:
        hw = hdiag.to(device).float().view(1, groups, g).expand(out_f, groups, g).reshape(out_f * groups, g)
    scale_list, zero_list = [], []
    chunk = 1 << 18
    for i in range(0, wg.shape[0], chunk):
        sub_w = wg[i : i + chunk]
        if hw is None:
            s, z = group_params(sub_w, spec)
        else:
            s, z = _group_params_rowweights(sub_w, spec, hw[i : i + chunk])
        scale_list.append(s)
        zero_list.append(z)
    s = torch.cat(scale_list).view(out_f, groups)
    z = torch.cat(zero_list).view(out_f, groups)
    q = torch.clamp(torch.round(w / s.repeat_interleave(g, 1)) + z.repeat_interleave(g, 1), 0, spec.maxq)
    return QResult(q.to(torch.uint8).cpu(), s.to(torch.bfloat16).cpu(), z.to(torch.uint8).cpu())


@torch.no_grad()
def _group_params_rowweights(w, spec, wt):
    maxq = spec.maxq
    xmin, xmax = w.min(1).values, w.max(1).values
    best_err = torch.full_like(xmin, float("inf"))
    best_s = torch.zeros_like(xmin)
    best_z = torch.zeros_like(xmin)
    for p in clip_grid(spec.bits):
        lo, hi = xmin * p, xmax * p
        s = ((hi - lo) / maxq).clamp(min=1e-10).to(torch.bfloat16).float()
        z = torch.round(-lo / s).clamp(0, maxq)
        q = torch.clamp(torch.round(w / s[:, None]) + z[:, None], 0, maxq)
        err = (((q - z[:, None]) * s[:, None] - w).pow(2) * wt).sum(1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
        best_z = torch.where(better, z, best_z)
    return best_s, best_z


@torch.no_grad()
def gptq(weight: torch.Tensor, H: torch.Tensor, spec: QSpec, damp: float = 0.01, blocksize: int = 128) -> QResult:
    """GPTQ with static groups in natural column order. weight [out, in] (any dtype), H [in, in] fp32 on GPU."""
    W = weight.to(device=H.device, dtype=torch.float32).clone()
    out_f, in_f = W.shape
    g = spec.group
    maxq = spec.maxq
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    hdiag = torch.diag(H).clone()
    H += damp * torch.mean(torch.diag(H)) * torch.eye(in_f, device=H.device)
    try:
        L = torch.linalg.cholesky(H)
    except Exception:  # noqa: BLE001
        H += 0.1 * torch.mean(torch.diag(H)) * torch.eye(in_f, device=H.device)
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    codes = torch.zeros(out_f, in_f, dtype=torch.uint8, device=H.device)
    scales = torch.zeros(out_f, in_f // g, device=H.device)
    zeros = torch.zeros(out_f, in_f // g, device=H.device)
    blocksize = max(g, (blocksize // g) * g)
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        s = z = None
        for i in range(count):
            col = i1 + i
            if col % g == 0:
                s, z = group_params(W1[:, i : i + g], spec, hdiag[col : col + g])
                scales[:, col // g] = s
                zeros[:, col // g] = z
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / s) + z, 0, maxq)
            codes[:, col] = q.to(torch.uint8)
            dq = (q - z) * s
            err = (w - dq) / d
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return QResult(codes.cpu(), scales.to(torch.bfloat16).cpu(), zeros.to(torch.uint8).cpu())


# ---------------------------------------------------------------------------------------------- layer pipeline


class _Stop(Exception):
    pass


def decoder_layers(model) -> nn.ModuleList:
    """The longest ModuleList of the model: its decoder blocks."""
    best = None
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and (best is None or len(module) > len(best)):
            best = module
    if best is None:
        raise RuntimeError("no decoder layer list found")
    return best


def module_names(model) -> dict[int, str]:
    return {id(m): n for n, m in model.named_modules()}


@torch.no_grad()
def capture_layer_inputs(model, layers, samples: list[torch.Tensor], device):
    """Per sample: (hidden_states, extra args, kwargs) as the first decoder layer receives them."""
    store = []

    def hook(_module, args, kwargs):
        args = list(args)
        if args:
            hidden = args.pop(0)
        else:
            hidden = kwargs.pop("hidden_states")
        kwargs = {k: v for k, v in kwargs.items() if k not in ("past_key_value", "past_key_values", "use_cache")}
        store.append((hidden, tuple(args), kwargs))
        raise _Stop

    handle = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for ids in samples:
            try:
                model(input_ids=ids.view(1, -1).to(device), use_cache=False)
            except _Stop:
                pass
    finally:
        handle.remove()
    return store


def _layer_out(out):
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def run_layer(layer, hidden, args, kwargs):
    return _layer_out(layer(hidden, *args, **kwargs))


@torch.no_grad()
def quantize_model_gptq(
    model,
    samples: list[torch.Tensor],
    spec: QSpec,
    damp: float = 0.01,
    true_sequential: bool = True,
    time_left=lambda: float("inf"),
) -> dict[str, QResult]:
    """Sequential GPTQ: each layer sees the outputs of the already-quantised layers before it.

    The model must sit on one CUDA device. Weights of quantised Linears are replaced in place by their dequantised
    values. Returns {checkpoint weight name: QResult}. Falls back to RTN for the remaining layers if time runs short.
    """
    device = next(model.parameters()).device
    layers = decoder_layers(model)
    names = module_names(model)
    t0 = time.time()
    store = capture_layer_inputs(model, layers, samples, device)
    hiddens = [s[0] for s in store]
    results: dict[str, QResult] = {}
    use_rtn = False
    for li, layer in enumerate(layers):
        tl = time.time()
        linears = {
            names[id(m)]: m
            for m in layer.modules()
            if isinstance(m, nn.Linear) and m.weight.shape[1] % spec.group == 0
        }
        remaining = dict(linears)
        while remaining:
            Hs = {n: None for n in remaining}
            counts = {n: 0 for n in remaining}
            order: list[str] = []
            first_input: dict[str, int] = {}
            handles = []

            def make_hook(name):
                def hook(_m, inp, _out):
                    x = inp[0].detach()
                    if name not in first_input:
                        first_input[name] = x.data_ptr()
                        order.append(name)
                    if use_rtn:
                        return
                    x2 = x.reshape(-1, x.shape[-1]).float()
                    h = x2.t().matmul(x2)
                    Hs[name] = h if Hs[name] is None else Hs[name].add_(h)
                    counts[name] += x2.shape[0]

                return hook

            if not use_rtn:
                for n, m in remaining.items():
                    handles.append(m.register_forward_hook(make_hook(n)))
                try:
                    for i, (_h0, args, kwargs) in enumerate(store):
                        run_layer(layer, hiddens[i], args, kwargs)
                finally:
                    for h in handles:
                        h.remove()
            if not order:
                order = list(remaining)
            if true_sequential and not use_rtn:
                ptr = first_input.get(order[0])
                stage = [n for n in order if first_input.get(n) == ptr]
            else:
                stage = list(remaining)
            for n in stage:
                m = remaining.pop(n)
                if use_rtn or Hs.get(n) is None:
                    res = rtn(m.weight.data, spec, device=device)
                else:
                    H = Hs[n] * (2.0 / max(1, counts[n]))
                    res = gptq(m.weight.data, H, spec, damp=damp)
                    Hs[n] = None
                m.weight.data.copy_(res.dequant(spec.group, m.weight.dtype).to(m.weight.device))
                results[n + ".weight"] = res
            for n in remaining:
                Hs[n] = None
        # propagate through the quantised layer
        for i, (hidden, args, kwargs) in enumerate(store):
            hiddens[i] = run_layer(layer, hiddens[i], args, kwargs)
        dt = time.time() - tl
        left_layers = len(layers) - li - 1
        if not use_rtn and dt * left_layers * 1.2 > time_left():
            log.warning("GPTQ out of time after layer %d (%.1fs/layer, %.0fs left): RTN for the rest", li, dt, time_left())
            use_rtn = True
        log.info("layer %d/%d quantised (%s) in %.1fs", li + 1, len(layers), "rtn" if use_rtn else "gptq", dt)
    torch.cuda.empty_cache()
    log.info("quantised %d linears in %.0fs", len(results), time.time() - t0)
    return results
