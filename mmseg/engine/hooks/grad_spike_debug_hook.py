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
        self._handles = []
        self._grad_items = []

    def before_train(self, runner):
        if self._handles:
            return

        for name, param in runner.model.named_parameters():
            if not param.requires_grad:
                continue

            def save_grad(grad, name=name, shape=tuple(param.shape)):
                norm = torch.linalg.vector_norm(grad.detach().float()).item()
                if not math.isfinite(norm):
                    norm = float('inf')
                self._grad_items.append((norm, name, shape))

            self._handles.append(param.register_hook(save_grad))

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._grad_items = []

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.interval > 1 and (runner.iter + 1) % self.interval != 0:
            return

        total_sq = 0.0
        for norm, _, _ in self._grad_items:
            total_sq += norm * norm

        if not self._grad_items:
            return
        total_norm = math.sqrt(total_sq)
        if math.isfinite(total_norm) and total_norm < self.threshold:
            return

        grad_items = sorted(
            self._grad_items, key=lambda item: item[0], reverse=True)
        summary = ', '.join(
            f'{name}: norm={norm:.4e}, shape={shape}'
            for norm, name, shape in grad_items[:self.topk])
        runner.logger.warning(
            f'[GradSpike] iter={runner.iter + 1} '
            f'total_norm={total_norm:.4e} top{self.topk}: {summary}')

    def after_train(self, runner):
        for handle in self._handles:
            handle.remove()
        self._handles = []
