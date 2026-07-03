"""Deploy CLIP-enhanced PVSA-Net by fusing all CLIP modules.

Usage:
    python tools/deploy_clip_pvsa.py \
        --config configs-h/clip/waterseg.py \
        --checkpoint work_dirs/clip_waterseg/best_mIoU.pth \
        --output work_dirs/deployed/

After deployment:
- Backbone text is frozen after TextRefiner
- Backbone TTRM/TextCrossAttention use pre-computed frozen K/V
- Head keeps image-conditioned prototype pooling by default
- Head fusion into Conv2d is only for the legacy fixed-prototype path
"""

import argparse
import os

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

    # Fuse all CLIP modules for deployment
    model.fuse_for_deployment()
    print('All CLIP modules fused for deployment')

    # Save deployed model
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
