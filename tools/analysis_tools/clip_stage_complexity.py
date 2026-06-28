import argparse
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope

import mmseg.models  # noqa: F401
from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile CLIP-PVSA model stage complexity.')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--device',
        default='cuda',
        help='cuda or cpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options')
    return parser.parse_args()


def _input_shape(shape):
    if len(shape) == 1:
        return 3, shape[0], shape[0]
    if len(shape) == 2:
        return 3, shape[0], shape[1]
    raise ValueError('invalid input shape')


def _count_params(module):
    return sum(p.numel() for p in module.parameters())


def _count_params_by_prefix(module, prefixes):
    total = 0
    for name, param in module.named_parameters():
        if any(name == prefix or name.startswith(prefix + '.')
               for prefix in prefixes):
            total += param.numel()
    return total


def _format(value):
    return f'{value / 1e6:.3f}M'


def _register_flops_hooks(module):
    flops_dict = {}
    hooks = []

    def _make_hook(name):
        def hook_fn(m, inp, out):
            flops = 0
            if isinstance(m, torch.nn.Linear):
                flops = 2 * inp[0].shape[0] * m.in_features * m.out_features
            elif isinstance(m, torch.nn.Conv2d):
                out_h, out_w = out.shape[2], out.shape[3]
                flops = 2 * m.in_channels * m.out_channels * \
                    m.kernel_size[0] * m.kernel_size[1] * out_h * out_w // m.groups
            elif isinstance(m, torch.nn.BatchNorm2d):
                flops = inp[0].numel() * 2
            if flops > 0:
                flops_dict[name] = flops
        return hook_fn

    for name, m in module.named_modules():
        hooks.append(m.register_forward_hook(_make_hook(name)))

    return flops_dict, hooks


def _sum_flops(flops_dict, prefixes):
    total = 0.0
    for name, flops in flops_dict.items():
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + '.'):
                total += flops
                break
    return total


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f'config file not found: {cfg_path}')

    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))
    if cfg.model.get('backbone', None) is not None:
        cfg.model.backbone.topp_flash_backend = None
        cfg.model.backbone.topp_flash_debug = False

    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')
    model = MODELS.build(cfg.model)
    model.eval().to(device)

    input_shape = _input_shape(args.shape)
    dummy = torch.randn(1, *input_shape, device=device)

    # ---- Backbone analysis ----
    backbone = model.backbone
    backbone.eval()
    if hasattr(backbone, '_disable_inference_fusion'):
        backbone._disable_inference_fusion = True

    bb_flops_dict, bb_hooks = _register_flops_hooks(backbone)
    with torch.no_grad():
        backbone(dummy)
    for h in bb_hooks:
        h.remove()

    # Stage-level breakdown
    fam_stages = set(getattr(backbone, 'fam_stages', (0, 1, 2, 3)))
    fusion_stages = set(getattr(backbone, 'fusion_stages', (0, 1, 2, 3)))
    ttrm_stages = set(getattr(backbone, 'ttrm_stages', ()))
    cross_attn_stages = set(getattr(backbone, 'cross_attn_stages', ()))

    print('=' * 80)
    print('Backbone Stage Complexity')
    print('=' * 80)
    print(f'{"stage":>5} | {"cnn":>16} | {"transformer":>16} | {"ttrm":>16} | '
          f'{"cross_attn":>16} | {"FAM":>16} | {"vote_fusion":>16} | {"out_norm":>16}')
    print('-' * 130)

    for stage in range(4):
        prefixes = {
            'cnn': [f'downsample_layers2.{stage}'],
            'transformer': [f'downsample_layers.{stage}', f'stages.{stage}'],
            'FAM': [f'FAM.{stage}'] if stage in fam_stages else [],
            'vote_fusion': [],
            'out_norm': [f'extra_norms.{stage}'],
        }
        if stage in fusion_stages:
            prefixes['vote_fusion'].extend([
                f'conv11.{stage}', f'conv12.{stage}',
                f'bn11.{stage}', f'bn12.{stage}',
            ])

        # TTRM: inside blocks, prefix is stages.{stage}.{block_idx}.PA.ttrm
        ttrm_prefixes = []
        if stage in ttrm_stages:
            depth = getattr(backbone, 'depth', [3, 4, 6, 3])[stage]
            for j in range(depth):
                ttrm_prefixes.append(f'stages.{stage}.{j}.PA.ttrm')

        # Cross-attention: stages.{stage}.{block_idx}.cross_attn
        ca_prefixes = []
        if stage in cross_attn_stages:
            depth = getattr(backbone, 'depth', [3, 4, 6, 3])[stage]
            for j in range(depth):
                ca_prefixes.append(f'stages.{stage}.{j}.cross_attn')

        cells = []
        for group in ('cnn', 'transformer', 'ttrm', 'cross_attn',
                       'FAM', 'vote_fusion', 'out_norm'):
            if group == 'ttrm':
                p = ttrm_prefixes
            elif group == 'cross_attn':
                p = ca_prefixes
            else:
                p = prefixes[group]
            gf = _sum_flops(bb_flops_dict, p)
            gp = _count_params_by_prefix(backbone, p)
            cells.append(f'{gf / 1e6:.2f}F/{_format(gp)}')
        print(f'{stage:>5} | ' + ' | '.join(cells))

    bb_total_params = _count_params(backbone)
    bb_total_flops = sum(bb_flops_dict.values())
    print(f'\nBackbone Total: {bb_total_flops / 1e9:.2f}G FLOPs, {bb_total_params / 1e6:.2f}M params')

    # ---- Text Encoder analysis ----
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        text_enc = model.text_encoder
        text_enc.eval()
        te_params = _count_params(text_enc)
        print('\n' + '=' * 80)
        print('Text Encoder')
        print('=' * 80)
        print(f'  Params: {_format(te_params)}')
        print(f'  Frozen: {not any(p.requires_grad for p in text_enc.parameters())}')

    # ---- Decode Head analysis ----
    if hasattr(model, 'decode_head') and model.decode_head is not None:
        head = model.decode_head
        head.eval()
        head_flops_dict, head_hooks = _register_flops_hooks(head)

        # Run forward to get FLOPs
        with torch.no_grad():
            feats = backbone(dummy)
            if isinstance(feats, tuple):
                feats = feats[0]
            if isinstance(feats, tuple):
                head(feats)
            else:
                head(feats)

        for h in head_hooks:
            h.remove()

        head_params = _count_params(head)
        head_flops = sum(head_flops_dict.values())
        print('\n' + '=' * 80)
        print('Decode Head (CLIPSegHead)')
        print('=' * 80)
        print(f'  Params: {_format(head_params)}')
        print(f'  FLOPs:  {head_flops / 1e6:.2f}M')

    # ---- Total ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n' + '=' * 80)
    print('Total Model')
    print('=' * 80)
    print(f'  Total params:     {_format(total_params)}')
    print(f'  Trainable params: {_format(trainable)}')
    print(f'  Frozen params:    {_format(total_params - trainable)}')


if __name__ == '__main__':
    main()
