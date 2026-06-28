import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class GradMonitorHook(Hook):
    """Monitor gradient norms for key parameter groups every iteration.

    Logs gradient norms for TTRM, text_encoder, backbone, and head
    into the training log to detect gradient vanishing/exploding.
    """

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        grad_norms = {}

        ttrm_grads = []
        text_grads = []
        backbone_grads = []
        head_grads = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad.norm().item()
            if 'ttrm' in name:
                ttrm_grads.append(g)
            elif 'text_encoder' in name:
                text_grads.append(g)
            elif 'decode_head' in name:
                head_grads.append(g)
            elif 'backbone' in name:
                backbone_grads.append(g)

        if ttrm_grads:
            grad_norms['g_ttrm'] = sum(ttrm_grads) / len(ttrm_grads)
        if text_grads:
            grad_norms['g_text'] = sum(text_grads) / len(text_grads)
        if backbone_grads:
            grad_norms['g_bb'] = sum(backbone_grads) / len(backbone_grads)
        if head_grads:
            grad_norms['g_head'] = sum(head_grads) / len(head_grads)

        # Write to message hub so they appear in the log
        for k, v in grad_norms.items():
            runner.message_hub.update_scalar(f'grad/{k}', v)
