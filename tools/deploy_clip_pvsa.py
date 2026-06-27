"""Deploy CLIP-enhanced PVSA-Net by fusing text components.

Usage:
    python tools/deploy_clip_pvsa.py \
        --config configs-h/biformer/biformer_clip_waterseg.py \
        --checkpoint work_dirs/best_mIoU.pth \
        --output work_dirs/deployed/

This script:
1. Loads trained model with CPFM-enhanced prototypes
2. Freezes prototypes and saves as .pt
3. Fuses BNContrastiveHead into Conv2d
4. Removes CPFM and TextEncoder modules
5. Saves deployment-ready model
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        description='Deploy CLIP-enhanced PVSA-Net')
    parser.add_argument('--config', type=str, required=True,
                        help='Model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Training checkpoint')
    parser.add_argument('--output', type=str, default='work_dirs/deployed/',
                        help='Output directory')
    parser.add_argument('--proto-save', type=str, default=None,
                        help='Path to save frozen prototypes')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Import mmseg
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)

    # Build model
    from mmseg.registry import MODELS
    model = MODELS.build(cfg.model)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load weights (allow missing keys for text_encoder)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    model.eval()

    # Step 1: Freeze prototypes
    proto_path = args.proto_save or os.path.join(args.output, 'frozen_prototypes.pt')
    model.freeze_prototypes(save_path=proto_path)
    print(f'Frozen prototypes saved to {proto_path}')

    # Step 2: Fuse decode head
    with torch.no_grad():
        model.decode_head.fuse_for_deployment(model.frozen_prototypes)
    print('Decode head fused into Conv2d')

    # Step 3: Remove CPFM and text gating modules from backbone
    if hasattr(model.backbone, 'cpfm_enabled') and model.backbone.cpfm_enabled:
        model.backbone.cpfm_modules = None
        model.backbone.cpfm_agg = None
        model.backbone.cpfm_enabled = False
        print('CPFM modules removed')
    if hasattr(model.backbone, 'use_gate_text') and model.backbone.use_gate_text:
        model.backbone.text_proj = None
        model.backbone.conv_text = None
        model.backbone.use_gate_text = False
        print('Text gating modules removed')

    # Step 4: Bake TTRM α into routing constant
    with torch.no_grad():
        for stage in model.backbone.stages:
            for block in stage:
                if hasattr(block, 'PA') and hasattr(block.PA, 'router'):
                    router = block.PA.router
                    if hasattr(router, 'use_ttrm') and router.use_ttrm:
                        alpha_val = torch.sigmoid(router.ttrm_alpha).item()
                        router._frozen_alpha = alpha_val
                        router.use_ttrm = False
                        print(f'  TTRM α fused: {alpha_val:.4f}')
    print('TTRM α fused into routing constants')

    # Step 5: Fuse RepRTA in text encoder
    model.text_encoder.fuse()
    print('Text encoder RepRTA fused')

    # Step 6: Save deployed model (save state_dict BEFORE removing modules)
    deployed_state = {
        'state_dict': model.state_dict(),
        'frozen_prototypes': model.frozen_prototypes,
        'config': cfg.text,
    }
    deployed_path = os.path.join(args.output, 'deployed_model.pth')
    torch.save(deployed_state, deployed_path)
    print(f'Deployed model saved to {deployed_path}')

    # Verify: run a dummy forward pass
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    print(f'Verification: output shape = {output.shape}')
    print('Deployment complete!')


if __name__ == '__main__':
    main()
