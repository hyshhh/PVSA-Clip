import torch
import torch.nn as nn


def attach_flops_hooks(model):
    """注册前向 hook，统计 FLOPs 与“实际触达”参数量。

    FLOPs 恢复上一版：只统计 Linear / Conv2d / BatchNorm2d，
    不额外估算注意力矩阵乘，避免数值虚高。

    参数量保持当前语义：仅累计前向真正触达的模块参数，
    关闭 FAM / VFM / CNN 时不计入闲置参数。
    """
    flops_dict = {}
    active_param_ids = set()
    hooks = []

    def _record_flops(name, flops):
        if flops <= 0:
            return
        flops_dict[name] = flops_dict.get(name, 0) + int(flops)

    def _record_params(module):
        for param in module.parameters(recurse=False):
            active_param_ids.add(id(param))

    def _make_hook(name):
        def hook_fn(module, inp, out):
            # 参数统计：凡是前向实际触达的模块都计入其自身参数。
            _record_params(module)

            if isinstance(module, nn.Linear):
                # 上一版：2 * batch * in_features * out_features
                # 对 Linear 输出，out.numel() = batch * out_features
                _record_flops(name, 2 * out.numel() * int(module.in_features))
                return

            if isinstance(module, nn.Conv2d):
                out_h, out_w = out.shape[2], out.shape[3]
                flops = (
                    2 * module.in_channels * module.out_channels *
                    module.kernel_size[0] * module.kernel_size[1] *
                    out_h * out_w // module.groups
                )
                _record_flops(name, flops)
                return

            if isinstance(module, nn.BatchNorm2d):
                _record_flops(name, inp[0].numel() * 2)
                return

        return hook_fn

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    return flops_dict, active_param_ids, hooks


def count_active_params(model, active_param_ids):
    """根据前向触达到的参数 id 统计有效参数量。"""
    total = 0
    for param in model.parameters():
        if id(param) in active_param_ids:
            total += param.numel()
    return int(total)


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
