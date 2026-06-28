import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradMonitorHook(Hook):
    """Monitor gradient norms for key parameter groups during training.

    Logs gradient norms for TTRM, text_encoder, backbone, and head parameters
    to detect gradient vanishing/exploding and monitor training dynamics.

    Args:
        interval (int): Log every N iterations. Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if (runner.iter + 1) % self.interval != 0:
            return

        model = runner.model
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model = model.module

        grad_norms = {}

        # TTRM parameters
        ttrm_grads = []
        for name, param in model.named_parameters():
            if 'ttrm' in name and param.grad is not None:
                ttrm_grads.append(param.grad.norm().item())
        if ttrm_grads:
            grad_norms['grad_ttrm'] = sum(ttrm_grads) / len(ttrm_grads)

        # Text encoder parameters
        text_grads = []
        for name, param in model.named_parameters():
            if 'text_encoder' in name and param.grad is not None:
                text_grads.append(param.grad.norm().item())
        if text_grads:
            grad_norms['grad_text_enc'] = sum(text_grads) / len(text_grads)

        # Backbone parameters (excluding TTRM)
        backbone_grads = []
        for name, param in model.named_parameters():
            if 'backbone' in name and 'ttrm' not in name and param.grad is not None:
                backbone_grads.append(param.grad.norm().item())
        if backbone_grads:
            grad_norms['grad_backbone'] = sum(backbone_grads) / len(backbone_grads)

        # Head parameters
        head_grads = []
        for name, param in model.named_parameters():
            if 'decode_head' in name and param.grad is not None:
                head_grads.append(param.grad.norm().item())
        if head_grads:
            grad_norms['grad_head'] = sum(head_grads) / len(head_grads)

        # Log via runner
        runner.logger.info(
            f'Grad norms: ' + ' '.join(f'{k}={v:.6f}' for k, v in grad_norms.items())
        )
