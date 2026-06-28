import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradMonitorHook(Hook):
    """Monitor gradient norm every N iterations.

    Args:
        interval (int): Log every N iterations. Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.interval != 0:
            return

        # Get grad norm from optim_wrapper (computed before optimizer.step)
        try:
            optim = runner.optim_wrapper
            if hasattr(optim, '_grad_norm'):
                norms = optim._grad_norm
                if norms is not None:
                    total = sum(n.item() ** 2 for n in norms.values()) ** 0.5
                    runner.logger.info(f'grad_norm={total:.4f}')
                    return
        except Exception:
            pass

        # Fallback: compute from model parameters (may be 0 if optimizer cleared)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        runner.logger.info(f'grad_norm={total_norm:.4f}')
