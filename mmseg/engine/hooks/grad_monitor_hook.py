import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradMonitorHook(Hook):
    """Monitor total gradient norm every iteration."""

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        runner.message_hub.update_scalar('grad/total_norm', total_norm)
