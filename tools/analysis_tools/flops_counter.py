import math

import torch
import torch.nn as nn

from mmseg.models.utils.bra_legacy import BiLevelRoutingAttention
from mmseg.models.utils.common import Attention, AttentionLePE
from mmseg.models.utils.top_p_bra import ToppAttention, TopkRouting as ToppTopkRouting


def _as_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            return default
        return int(value[0])
    return int(value)


def _get_padded_hw(height, width, n_win, auto_pad):
    if not auto_pad:
        return int(height), int(width)
    padded_h = int(math.ceil(height / n_win) * n_win)
    padded_w = int(math.ceil(width / n_win) * n_win)
    return padded_h, padded_w


def _estimate_kv_tokens(module, local_h, local_w):
    mode = getattr(module, 'kv_downsample_mode', 'identity')
    if mode in ('identity', None):
        return int(local_h * local_w)
    if mode in ('avgpool', 'maxpool'):
        ratio = max(1, _as_int(getattr(module, 'kv_downsample_ratio', 1), default=1))
        return int(math.ceil(local_h / ratio) * math.ceil(local_w / ratio))
    if mode in ('ada_avgpool', 'ada_maxpool'):
        kv_per_win = getattr(module, 'kv_per_win', local_h)
        if isinstance(kv_per_win, (tuple, list)):
            if len(kv_per_win) == 2:
                return int(kv_per_win[0]) * int(kv_per_win[1])
            kv_per_win = kv_per_win[0]
        kv_per_win = _as_int(kv_per_win, default=local_h)
        return int(kv_per_win * kv_per_win)
    raise NotImplementedError(f'Unsupported kv_downsample_mode for FLOPs counting: {mode}')


def _estimate_topp_keep_len(module):
    keep_len = getattr(module.router, '_flops_keep_len', None)
    if keep_len is not None:
        return max(1.0, float(keep_len))
    maxk = float(getattr(module.router, 'topk', getattr(module, 'topk', 1)))
    keep_p = float(getattr(module.router, 'P', 1.0))
    mink = float(getattr(module.router, 'mink', 1))
    return max(mink, math.ceil(maxk * keep_p))


def _attention_extra_flops(module, x):
    if x.dim() != 4:
        return 0
    batch, height, width, _ = x.shape
    padded_h, padded_w = _get_padded_hw(
        height, width, module.n_win, getattr(module, 'auto_pad', False))
    num_windows = int(module.n_win * module.n_win)
    local_h = int(padded_h // module.n_win)
    local_w = int(padded_w // module.n_win)
    local_tokens = int(local_h * local_w)
    kv_tokens = _estimate_kv_tokens(module, local_h, local_w)
    if isinstance(module, ToppAttention):
        effective_topk = _estimate_topp_keep_len(module)
    else:
        effective_topk = float(getattr(module, 'topk', getattr(module.router, 'topk', 1)))

    routing_flops = 2 * batch * num_windows * num_windows * int(module.qk_dim)
    attention_flops = (
        2 * batch * num_windows * local_tokens * effective_topk * kv_tokens *
        (int(module.qk_dim) + int(module.dim))
    )
    return int(routing_flops + attention_flops)


def _plain_attention_extra_flops(module, x):
    if x.dim() != 4:
        return 0
    batch, height, width, channels = x.shape
    num_tokens = int(height * width)
    qk_flops = 2 * batch * num_tokens * num_tokens * int(channels)
    av_flops = 2 * batch * num_tokens * num_tokens * int(channels)
    return int(qk_flops + av_flops)


def attach_flops_hooks(model):
    flops_dict = {}
    hooks = []

    def _record(name, flops):
        if flops <= 0:
            return
        flops_dict[name] = flops_dict.get(name, 0) + int(flops)

    def _make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(module, ToppTopkRouting):
                if isinstance(out, tuple) and len(out) >= 3 and out[2] is not None:
                    valid_mask = out[2]
                    module._flops_keep_len = float(
                        valid_mask.to(dtype=torch.float32).sum(dim=-1).mean().item())
                else:
                    module._flops_keep_len = float(getattr(module, 'topk', 1))
                return

            if isinstance(module, nn.Linear):
                _record(name, 2 * out.numel() * int(module.in_features))
                return

            if isinstance(module, nn.Conv2d):
                kernel_h, kernel_w = module.kernel_size
                per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
                _record(name, 2 * out.numel() * per_output)
                return

            if isinstance(module, nn.BatchNorm2d):
                _record(name, out.numel() * 2)
                return

            if isinstance(module, (ToppAttention, BiLevelRoutingAttention)):
                _record(name, _attention_extra_flops(module, inp[0]))
                return

            if isinstance(module, (Attention, AttentionLePE)):
                _record(name, _plain_attention_extra_flops(module, inp[0]))

        return hook_fn

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    return flops_dict, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
