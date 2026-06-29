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
        description='Profile CLIP-PVSA model inference complexity.')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape', type=int, nargs='+', default=[256, 256],
        help='input image size')
    parser.add_argument(
        '--device', default='cuda', help='cuda or cpu')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
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


def _fmt(v):
    return f'{v / 1e6:.3f}M'


def _gf(v):
    return f'{v / 1e9:.3f}'


def _register_module_hooks(module):
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
    H, W = input_shape[1], input_shape[2]

    bb_cfg = cfg.model.backbone
    embed_dim = bb_cfg.embed_dim
    depth = bb_cfg.depth
    mlp_ratios = bb_cfg.mlp_ratios
    n_win = bb_cfg.n_win
    topks = bb_cfg.topks
    qk_dims = bb_cfg.qk_dims
    head_dim = bb_cfg.head_dim
    use_ttrm = bb_cfg.get('use_ttrm', False)
    ttrm_stages = set(bb_cfg.get('ttrm_stages', []))
    cross_attn_stages = set(bb_cfg.get('cross_attn_stages', []))
    num_classes = cfg.model.decode_head.get('num_classes', 3)
    text_dim = cfg.model.text_encoder.get('embed_dim', 512)
    win_size = n_win * n_win
    route_cfgs = bb_cfg.get('topp_route_configs', {})

    # ---- Method 1: Module hooks (Linear/Conv/BN only, comparable to get_flops.py) ----
    dummy = torch.randn(1, *input_shape, device=device)
    model.backbone._disable_inference_fusion = True

    # Run full model forward to get module-level FLOPs
    module_flops_dict, module_hooks = _register_module_hooks(model)
    with torch.no_grad():
        # Simulate inference path
        if hasattr(model, 'text_encoder') and model.text_encoder is not None:
            model._prototypes_frozen = True
            model.frozen_prototypes.copy_(
                torch.randn(num_classes, text_dim, device=device))
        model(dummy)
    for h in module_hooks:
        h.remove()
    module_total = sum(module_flops_dict.values())

    # ---- Method 2: Analytical (includes attention matmul) ----
    total_attn_matmul = 0
    total_ttrm_matmul = 0
    total_ca_matmul = 0

    print('=' * 95)
    print(f'Input: {input_shape}, window: {n_win}x{n_win}={win_size}')
    print('=' * 95)
    print(f'{"stage":>5} | {"module_flops":>13} | {"+attn_matmul":>13} | '
          f'{"+ttrm_matmul":>13} | {"+ca_matmul":>13} | {"stage_total":>13}')
    print('-' * 95)

    for stage in range(4):
        d = embed_dim[stage]
        qk = qk_dims[stage]
        n_heads = qk // head_dim
        topk = topks[stage]
        n_layers = depth[stage]
        h = H // (2 ** (stage + 2))
        w = W // (2 ** (stage + 2))
        n_windows = (h // n_win) * (w // n_win)
        n_tokens = h * w

        # Top-P effective topk
        p = route_cfgs.get(topk, {}).get('p', 0.5)
        effective_topk = max(1, int(topk * p))

        # Module FLOPs for this stage (from hooks)
        stage_prefixes = [f'backbone.trans_downsample_layers.{stage}',
                          f'backbone.stages.{stage}',
                          f'backbone.extra_norms.{stage}',
                          f'backbone.trans_conv.{stage}',
                          f'backbone.bn.{stage}']
        stage_module_flops = 0
        for name, flops in module_flops_dict.items():
            for prefix in stage_prefixes:
                if name.startswith(prefix):
                    stage_module_flops += flops
                    break

        # Attention matmul: Q@K.T + attn@V
        attn_matmul = (2 * 2 * n_windows * n_heads * win_size * head_dim *
                       effective_topk * win_size) * n_layers
        total_attn_matmul += attn_matmul

        # TTRM matmul: q@tc_k.T + k@tc_k.T + q_text@k_text.T
        ttrm_matmul = 0
        if use_ttrm and stage in ttrm_stages:
            K = num_classes
            ttrm_per_win = (2 * win_size * qk * K +   # q@tc_k.T
                            2 * win_size * qk * K +   # k@tc_k.T
                            2 * win_size * K * win_size)  # q_text@k_text.T
            ttrm_matmul = ttrm_per_win * n_windows * n_layers
        total_ttrm_matmul += ttrm_matmul

        # Cross-attention matmul: q@k.T + attn@v
        ca_matmul = 0
        if stage in cross_attn_stages:
            K = num_classes
            ca_matmul = (2 * n_tokens * d * K * 2) * n_layers  # q@k.T + attn@v
        total_ca_matmul += ca_matmul

        stage_total = stage_module_flops + attn_matmul + ttrm_matmul + ca_matmul
        print(f'{stage:>5} | {_gf(stage_module_flops):>11}G | '
              f'{_gf(attn_matmul):>11}G | {_gf(ttrm_matmul):>11}G | '
              f'{_gf(ca_matmul):>11}G | {_gf(stage_total):>11}G')

    # ---- Head FLOPs (from hooks) ----
    head_flops = 0
    for name, flops in module_flops_dict.items():
        if 'decode_head' in name:
            head_flops += flops

    total_analytical = (module_total + total_attn_matmul +
                        total_ttrm_matmul + total_ca_matmul)

    print(f'\n{"=" * 95}')
    print(f'Module FLOPs (Linear/Conv/BN):  {_gf(module_total)}G  '
          f'(comparable to get_flops.py)')
    print(f'  + Attention matmul:           {_gf(total_attn_matmul)}G')
    print(f'  + TTRM matmul:                {_gf(total_ttrm_matmul)}G')
    print(f'  + Cross-Attn matmul:          {_gf(total_ca_matmul)}G')
    print(f'Total Inference FLOPs:          {_gf(total_analytical)}G')
    print(f'Total Params: {_fmt(_count_params(model))}')


if __name__ == '__main__':
    main()
