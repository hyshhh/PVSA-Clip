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


def _flops(v):
    return f'{v / 1e6:.2f}'


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

    # Backbone config
    bb_cfg = cfg.model.backbone
    embed_dim = bb_cfg.embed_dim  # [64, 128, 256, 512]
    depth = bb_cfg.depth          # [3, 4, 6, 3]
    mlp_ratios = bb_cfg.mlp_ratios  # [3, 3, 3, 3]
    n_win = bb_cfg.n_win          # 7
    topks = bb_cfg.topks          # [16, 12, 8, 6]
    qk_dims = bb_cfg.qk_dims      # [64, 128, 256, 512]
    head_dim = bb_cfg.head_dim    # 32
    use_ttrm = bb_cfg.get('use_ttrm', False)
    ttrm_stages = set(bb_cfg.get('ttrm_stages', []))
    cross_attn_stages = set(bb_cfg.get('cross_attn_stages', []))
    num_classes = cfg.model.decode_head.get('num_classes', 3)
    text_dim = cfg.model.text_encoder.get('embed_dim', 512)
    win_size = n_win * n_win  # 49

    print('=' * 90)
    print(f'Input: {input_shape}, window: {n_win}x{n_win}={win_size}, '
          f'num_classes: {num_classes}')
    print('=' * 90)

    # Analytical FLOPs calculation per stage
    total_bb_flops = 0
    print(f'{"stage":>5} | {"transformer":>14} | {"ttrm":>14} | '
          f'{"cross_attn":>14} | {"vote_fusion":>14} | {"total":>14}')
    print('-' * 90)

    for stage in range(4):
        d = embed_dim[stage]
        qk = qk_dims[stage]
        n_heads = qk // head_dim
        topk = topks[stage]
        n_layers = depth[stage]
        mlp_ratio = mlp_ratios[stage]
        # Spatial resolution at this stage
        h = H // (2 ** (stage + 2))  # stem downsamples 4x, each stage 2x
        w = W // (2 ** (stage + 2))
        n_windows = (h // n_win) * (w // n_win)  # number of windows
        n_tokens = h * w  # total tokens

        # --- Transformer block FLOPs ---
        # 1. pos_embed (dwconv before attn): 2 * d * k^2 * n_tokens
        pos_embed_flops = 2 * d * 3 * 3 * n_tokens

        # 2. QKV projection: 3 * (2 * d * qk) * n_tokens
        qkv_flops = 3 * 2 * d * qk * n_tokens

        # 3. Window attention with topk routing:
        #    Q,K per window: [win_size, head_dim]
        #    attn = Q @ K.T: [win_size, topk * win_size] per window
        #    out = attn @ V: same
        attn_flops = 2 * n_windows * n_heads * win_size * head_dim * topk * win_size * 2

        # 4. Output projection: 2 * qk * d * n_tokens
        out_proj_flops = 2 * qk * d * n_tokens

        # 5. MLP: 2 * (d * mlp_d + mlp_d * d) * n_tokens
        mlp_d = int(mlp_ratio * d)
        mlp_flops = 2 * 2 * d * mlp_d * n_tokens

        # 6. LayerNorm (2 per block): ~5 * d * n_tokens each
        ln_flops = 2 * 5 * d * n_tokens

        transformer_flops = (pos_embed_flops + qkv_flops + attn_flops +
                             out_proj_flops + mlp_flops + ln_flops) * n_layers

        # --- TTRM FLOPs ---
        ttrm_flops = 0
        if use_ttrm and stage in ttrm_stages:
            K = num_classes
            # Per window: q @ tc_k.T [win_size, qk] @ [qk, K] = [win_size, K]
            # k @ tc_k.T same
            # q_text @ k_text.T [win_size, K] @ [K, win_size] = [win_size, win_size]
            ttrm_per_window = (2 * win_size * qk * K +   # q @ tc_k.T
                               2 * win_size * qk * K +   # k @ tc_k.T
                               2 * win_size * K * win_size)  # q_text @ k_text.T
            ttrm_flops = ttrm_per_window * n_windows * n_layers

        # --- Cross-attention FLOPs ---
        ca_flops = 0
        if stage in cross_attn_stages:
            K = num_classes
            # Per token: q @ k.T [n_tokens, d] @ [d, K] = [n_tokens, K]
            # attn @ v [n_tokens, K] @ [K, d] = [n_tokens, d]
            ca_per_token = 2 * n_tokens * d * K * 2  # q@k.T + attn@v
            ca_out_proj = 2 * d * d * n_tokens
            ca_flops = (ca_per_token + ca_out_proj) * n_layers

        # --- Vote fusion FLOPs ---
        vote_flops = 0
        if stage < 3:
            # conv11: Conv2d(d_next, d, 1x1) applied to upsampled feature
            h_cur = h
            w_cur = w
            vote_flops = 2 * embed_dim[stage + 1] * d * h_cur * w_cur
            # BN + sigmoid + upsample are negligible

        stage_total = transformer_flops + ttrm_flops + ca_flops + vote_flops
        total_bb_flops += stage_total

        print(f'{stage:>5} | {_flops(transformer_flops):>12}F | '
              f'{_flops(ttrm_flops):>12}F | {_flops(ca_flops):>12}F | '
              f'{_flops(vote_flops):>12}F | {_flops(stage_total):>12}F')

    print(f'\nBackbone Total: {total_bb_flops / 1e9:.2f}G FLOPs, '
          f'{_fmt(_count_params(model.backbone))} params')

    # ---- Text Encoder (removed after fusion) ----
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        te_params = _count_params(model.text_encoder)
        print(f'\nText Encoder (removed after fusion): {_fmt(te_params)} params')

    # ---- Decode Head ----
    if hasattr(model, 'decode_head') and model.decode_head is not None:
        head = model.decode_head
        head_params = _count_params(head)
        # Fused head: single Conv2d(channels, num_classes, 1x1)
        # Input: [B, 256, H/4, W/4] -> [B, num_classes, H/4, W/4]
        h4 = H // 4
        w4 = W // 4
        channels = cfg.model.decode_head.get('channels', 256)
        head_flops = 2 * channels * num_classes * h4 * w4
        print(f'\nDecode Head (CLIPSegHead, fused Conv2d):')
        print(f'  Params: {_fmt(head_params)} (before fusion)')
        print(f'  FLOPs:  {_flops(head_flops)}F (after fusion: single Conv2d)')

    # ---- Total ----
    total_params = _count_params(model)
    total_inference_flops = total_bb_flops + head_flops
    print(f'\n{"=" * 90}')
    print(f'Total Inference FLOPs: {total_inference_flops / 1e9:.2f}G')
    print(f'Total Params: {_fmt(total_params)} (TextEncoder excluded after fusion)')


if __name__ == '__main__':
    main()
