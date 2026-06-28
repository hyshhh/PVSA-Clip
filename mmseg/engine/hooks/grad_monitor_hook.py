import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradMonitorHook(Hook):
    """Monitor gradient norm after backward, before optimizer step.

    Args:
        interval (int): Log every N iterations. Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_backward(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.interval != 0:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        runner.logger.info(f'grad_norm={total_norm:.4f}')
