"""Deploy CLIP-enhanced PVSA-Net by pre-computing text projections.

Usage:
    python tools/deploy_clip_pvsa.py \
        --config configs-h/biformer/biformer_clip_waterseg.py \
        --checkpoint work_dirs/clip_waterseg/best_mIoU.pth \
        --output work_dirs/deployed/

This script:
1. Freezes category prototypes (original CLIP embeddings)
2. Pre-computes TTRM text projections (tc_k, tc_v) as fixed constants
3. Pre-computes TextCrossAttention K/V projections as fixed constants
4. Fuses BNContrastiveHead into Conv2d
5. RepRTA skipped (_fused=True)
6. Saves deployment-ready model (no TextEncoder needed)
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

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)

    from mmseg.registry import MODELS
    model = MODELS.build(cfg.model)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    model.eval()

    # Step 1: Freeze prototypes
    proto_path = args.proto_save or os.path.join(args.output, 'frozen_prototypes.pt')
    model.freeze_prototypes(save_path=proto_path)
    print(f'Frozen prototypes saved to {proto_path}')

    # Step 2: Fuse decode head (BN + proj + cosine → Conv2d)
    with torch.no_grad():
        model.decode_head.fuse_for_deployment(model.frozen_prototypes)
    print('Decode head fused into Conv2d')

    # Step 3: Pre-compute TTRM text projections
    with torch.no_grad():
        prototypes = model.frozen_prototypes  # [K, D]
        for stage in model.backbone.stages:
            for block in stage:
                if hasattr(block, 'PA') and hasattr(block.PA, 'router'):
                    router = block.PA.router
                    if hasattr(router, 'use_ttrm') and router.use_ttrm:
                        # Pre-compute text K/V for TTRM
                        tc = router.ttrm_norm(prototypes)
                        tc_k = torch.nn.functional.normalize(
                            router.ttrm_text_proj(tc), dim=-1)
                        tc_v = router.ttrm_text_v_proj(tc)
                        router.register_buffer('_frozen_tc_k', tc_k)
                        router.register_buffer('_frozen_tc_v', tc_v)
                        router._ttrm_precomputed = True
                        router.use_ttrm = False
                        print(f'  TTRM text projections pre-computed')
    print('TTRM text projections frozen')

    # Step 4: Pre-compute TextCrossAttention K/V
    with torch.no_grad():
        prototypes = model.frozen_prototypes
        for stage in model.backbone.stages:
            for block in stage:
                if hasattr(block, 'cross_attn') and block.cross_attn is not None:
                    ca = block.cross_attn
                    k = ca.text_proj_k(prototypes)  # [K, C]
                    # Merge out_proj into V: out_proj(v @ attn.T) = (out_proj(v)) @ attn.T
                    v = ca.out_proj(ca.text_proj_v(prototypes))  # [K, C]
                    ca.register_buffer('_frozen_k', k)
                    ca.register_buffer('_frozen_v', v)
                    ca._precomputed = True
        print('TextCrossAttention K/V pre-computed')

    # Step 5: Fuse RepRTA
    model.text_encoder.fuse()
    print('Text encoder RepRTA fused')

    # Step 6: Save deployed model
    deployed_state = {
        'state_dict': model.state_dict(),
        'frozen_prototypes': model.frozen_prototypes,
        'config': cfg.text,
    }
    deployed_path = os.path.join(args.output, 'deployed_model.pth')
    torch.save(deployed_state, deployed_path)
    print(f'Deployed model saved to {deployed_path}')

    # Verify
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(dummy_input)
    print(f'Verification: output shape = {output.shape}')
    print('Deployment complete!')


if __name__ == '__main__':
    main()
