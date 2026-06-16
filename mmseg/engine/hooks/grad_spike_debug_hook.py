# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradSpikeDebugHook(Hook):
    """Log parameters with the largest gradients when a spike appears."""

    def __init__(self, threshold=1e5, topk=10, interval=1):
        self.threshold = float(threshold)
        self.topk = int(topk)
        self.interval = int(interval)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.interval > 1 and (runner.iter + 1) % self.interval != 0:
            return

        grad_items = []
        total_sq = 0.0
        has_grad = False
        for name, param in runner.model.named_parameters():
            if param.grad is None:
                continue
            has_grad = True
            grad = param.grad.detach()
            norm = torch.linalg.vector_norm(grad.float()).item()
            if not math.isfinite(norm):
                norm = float('inf')
            grad_items.append((norm, name, tuple(param.shape)))
            total_sq += norm * norm

        if not has_grad:
            return
        total_norm = math.sqrt(total_sq)
        if math.isfinite(total_norm) and total_norm < self.threshold:
            return

        grad_items.sort(key=lambda item: item[0], reverse=True)
        summary = ', '.join(
            f'{name}: norm={norm:.4e}, shape={shape}'
            for norm, name, shape in grad_items[:self.topk])
        runner.logger.warning(
            f'[GradSpike] iter={runner.iter + 1} '
            f'total_norm={total_norm:.4e} top{self.topk}: {summary}')
