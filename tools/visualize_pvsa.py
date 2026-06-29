import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from mmengine.config import DictAction
except ImportError:
    class DictAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            options = {}
            for item in values:
                if '=' not in item:
                    parser.error(f'Invalid cfg option: {item}')
                key, value = item.split('=', 1)
                options[key] = value
            setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize PVSA feature maps and Top-P routing maps.')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        'test_index',
        nargs='?',
        type=int,
        help='1-based image index in config test dataset')
    parser.add_argument('--image', help='input image path')
    parser.add_argument(
        '--test-index',
        dest='test_index_opt',
        metavar='TEST_INDEX',
        type=int,
        help='1-based image index in config test dataset')
    parser.add_argument(
        '--mode',
        choices=['baseline', 'clip'],
        required=True,
        help='visualization path')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument(
        '--feature-dir',
        default='demo/feathermap',
        help='feature map output directory')
    parser.add_argument(
        '--attn-dir',
        default='demo/attension_map',
        help='attention map output directory')
    parser.add_argument(
        '--query-index',
        type=int,
        default=32,
        help='query window index for route visualization')
    parser.add_argument(
        '--route-stage',
        type=int,
        default=0,
        help='route stage to visualize when --single-route is set')
    parser.add_argument(
        '--route-block',
        type=int,
        default=0,
        help='route block to visualize when --single-route is set')
    parser.add_argument(
        '--all-routes',
        action='store_true',
        default=True,
        help='save route maps for all routed blocks, enabled by default')
    parser.add_argument(
        '--single-route',
        action='store_true',
        help='only save the route selected by --route-stage and --route-block')
    parser.add_argument(
        '--feature-reduce',
        choices=['l2', 'mean', 'max'],
        default='l2',
        help='channel reduction for feature maps')
    parser.add_argument(
        '--dark-ratio',
        type=float,
        default=0.3,
        help='dark ratio for unselected route windows')
    parser.add_argument(
        '--keep-cuda-route',
        action='store_true',
        help='keep configured CUDA route backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options')
    return parser.parse_args()


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def image_stem(path):
    return osp.splitext(osp.basename(path))[0]


def resolve_test_image_from_config(cfg, test_index):
    if test_index is None:
        return None
    if test_index <= 0:
        raise ValueError('--test-index uses 1-based indexing and must be >= 1.')
    if 'test_dataloader' not in cfg:
        raise KeyError('test_dataloader is not found in config.')

    from mmseg.registry import DATASETS

    dataset_cfg = cfg.test_dataloader.get('dataset')
    if dataset_cfg is None:
        raise KeyError('test_dataloader.dataset is not found in config.')
    dataset = DATASETS.build(dataset_cfg)
    zero_index = test_index - 1
    if zero_index >= len(dataset):
        raise IndexError(
            f'test_index={test_index} exceeds test dataset size={len(dataset)}.')

    if hasattr(dataset, 'get_data_info'):
        data_info = dataset.get_data_info(zero_index)
    elif hasattr(dataset, 'data_list'):
        data_info = dataset.data_list[zero_index]
    else:
        raise TypeError('Cannot read image path from this dataset object.')

    img_path = data_info.get('img_path')
    if img_path is None and 'img_path' in data_info.get('data_samples', {}):
        img_path = data_info['data_samples']['img_path']
    if img_path is None:
        raise KeyError('img_path is not found in selected test data.')
    print(f'Use test dataset image #{test_index}: {img_path}')
    return img_path


def sanitize_name(name):
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', name).strip('_')


def tensor_to_chw(feat, layout='nchw'):
    if isinstance(feat, (tuple, list)):
        feat = feat[0]
    feat = feat.detach().float().cpu()
    if feat.dim() == 4:
        feat = feat[0]
        if layout == 'nhwc':
            feat = feat.permute(2, 0, 1)
    elif feat.dim() == 3 and layout == 'hwc':
        feat = feat.permute(2, 0, 1)
    if feat.dim() != 3:
        raise ValueError(f'Unsupported feature shape: {tuple(feat.shape)}')
    return feat


def feature_to_map(feat, layout='nchw', reduce='l2'):
    feat = tensor_to_chw(feat, layout=layout)
    if reduce == 'mean':
        fmap = feat.mean(dim=0)
    elif reduce == 'max':
        fmap = feat.max(dim=0).values
    else:
        fmap = torch.linalg.vector_norm(feat, ord=2, dim=0)
    fmap = fmap.numpy()
    low, high = np.percentile(fmap, [1, 99])
    fmap = np.clip(fmap, low, high)
    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
    fmap = cv2.GaussianBlur(fmap, (7, 7), 0)
    return fmap


def upsample_map(fmap, target_hw):
    tensor = torch.from_numpy(fmap).float()[None, None]
    tensor = F.interpolate(
        tensor, size=target_hw, mode='bilinear', align_corners=False)
    return tensor[0, 0].numpy()


def apply_colormap(fmap, colormap=cv2.COLORMAP_VIRIDIS):
    return cv2.applyColorMap((fmap * 255).astype(np.uint8), colormap)


def save_feature_images(feat, img_bgr, save_prefix, layout='nchw', reduce='l2'):
    h, w = img_bgr.shape[:2]
    fmap = feature_to_map(feat, layout=layout, reduce=reduce)
    fmap = upsample_map(fmap, (h, w))
    heatmap = apply_colormap(fmap)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    mkdir(osp.dirname(save_prefix))
    cv2.imwrite(f'{save_prefix}_heatmap.png', heatmap)
    cv2.imwrite(f'{save_prefix}_overlay.png', overlay)
    return fmap


def save_diff_images(before, after, img_bgr, save_prefix, layout='nhwc'):
    before = tensor_to_chw(before, layout=layout)
    after = tensor_to_chw(after, layout=layout)
    diff = torch.linalg.vector_norm(after - before, ord=2, dim=0).numpy()
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
    diff = cv2.GaussianBlur(diff, (7, 7), 0)
    diff = upsample_map(diff, img_bgr.shape[:2])
    heatmap = apply_colormap(diff, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
    cv2.imwrite(f'{save_prefix}_diff_heatmap.png', heatmap)
    cv2.imwrite(f'{save_prefix}_diff_overlay.png', overlay)


def register_stage_hooks(backbone, feature_store):
    handles = []
    if not hasattr(backbone, 'stages'):
        return handles
    for stage_idx, stage in enumerate(backbone.stages):
        target = stage[-1] if len(stage) > 0 else stage

        def hook(_module, _inputs, output, stage_idx=stage_idx):
            feature_store[f'stage{stage_idx + 1}_output'] = output
        handles.append(target.register_forward_hook(hook))
    return handles


def register_text_hooks(backbone, text_store):
    handles = []
    for name, module in backbone.named_modules():
        if module.__class__.__name__ != 'TextCrossAttention':
            continue
        clean_name = sanitize_name(name)

        def pre_hook(_module, inputs, clean_name=clean_name):
            text_store.setdefault(clean_name, {})['before'] = inputs[0]

        def post_hook(_module, _inputs, output, clean_name=clean_name):
            text_store.setdefault(clean_name, {})['after'] = output

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def disable_cuda_route_backend(model):
    for module in model.modules():
        if hasattr(module, 'topp_flash_backend'):
            module.topp_flash_backend = None


def enable_route_debug_cache(model):
    for module in model.modules():
        if hasattr(module, '_enable_route_debug_cache'):
            module._enable_route_debug_cache = True


def enable_existing_baseline_feature_vis(backbone, save_dir, target_hw):
    if not hasattr(backbone, 'feature_vis_config'):
        return
    backbone.feature_vis_config = dict(
        enabled=True,
        once=False,
        save_dir=save_dir,
        out_size=target_hw,
        channel_reduce='mean')
    if hasattr(backbone, '_feature_vis_saved'):
        backbone._feature_vis_saved = False


def parse_stage_block(name):
    match = re.search(r'stages\.(\d+)\.(\d+)\.PA\.router', name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def collect_route_modules(model, args):
    routes = []
    for name, module in model.named_modules():
        route_debug = getattr(module, '_last_route_debug', None)
        if route_debug is None:
            continue
        stage_idx, block_idx = parse_stage_block(name)
        if args.single_route:
            if stage_idx != args.route_stage or block_idx != args.route_block:
                continue
        routes.append((name, module, route_debug, stage_idx, block_idx))
    return routes


def block_bounds(index, grid, h, w):
    row = index // grid
    col = index % grid
    y0 = int(round(row * h / grid))
    y1 = int(round((row + 1) * h / grid))
    x0 = int(round(col * w / grid))
    x1 = int(round((col + 1) * w / grid))
    return x0, y0, x1, y1


def draw_grid(img, grid, color=(255, 255, 255)):
    h, w = img.shape[:2]
    for i in range(1, grid):
        x = int(round(i * w / grid))
        y = int(round(i * h / grid))
        cv2.line(img, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)


def put_centered_text(img, text, x0, y0, x1, y1, bg_value):
    cell_w = max(1, x1 - x0)
    cell_h = max(1, y1 - y0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.45, max(0.22, min(cell_w, cell_h) / 90.0))
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw > cell_w - 4:
        scale = max(0.18, scale * (cell_w - 4) / max(tw, 1))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = x0 + max(1, (cell_w - tw) // 2)
    y = y0 + max(th + 1, (cell_h + th) // 2)
    color = (0, 0, 0) if bg_value > 0.62 else (255, 255, 255)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def score_text(value):
    if abs(value) >= 1:
        return f'{value:.2f}'
    return f'{value:.3f}'


def draw_route_score_overlay(img_bgr, scores):
    h, w = img_bgr.shape[:2]
    total = int(scores.shape[0])
    grid = int(round(total ** 0.5))
    if grid * grid != total:
        raise ValueError(f'Route window count is not square: {total}')
    scores = scores.astype(np.float32)
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    block_map = np.zeros((h, w), dtype=np.float32)
    for idx, value in enumerate(norm):
        x0, y0, x1, y1 = block_bounds(idx, grid, h, w)
        block_map[y0:y1, x0:x1] = value
    heatmap = cv2.applyColorMap(
        (block_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
    draw_grid(overlay, grid)
    for idx, value in enumerate(scores):
        x0, y0, x1, y1 = block_bounds(idx, grid, h, w)
        put_centered_text(
            overlay, score_text(float(value)), x0, y0, x1, y1, norm[idx])
    return overlay


def draw_top_p_mask(img_bgr, selected_indices, selected_scores, total, dark_ratio):
    h, w = img_bgr.shape[:2]
    grid = int(round(total ** 0.5))
    if grid * grid != total:
        raise ValueError(f'Route window count is not square: {total}')
    selected = set(int(idx) for idx in selected_indices)
    dark = (img_bgr.astype(np.float32) * float(dark_ratio)).astype(np.uint8)
    overlay = dark.copy()
    for idx in selected:
        x0, y0, x1, y1 = block_bounds(idx, grid, h, w)
        overlay[y0:y1, x0:x1] = img_bgr[y0:y1, x0:x1]
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1),
                      (0, 255, 120), 2, cv2.LINE_AA)
    draw_grid(overlay, grid, color=(120, 120, 120))
    for idx, score in zip(selected_indices, selected_scores):
        x0, y0, x1, y1 = block_bounds(int(idx), grid, h, w)
        put_centered_text(
            overlay, score_text(float(score)), x0, y0, x1, y1, 1.0)
    return overlay


def save_route_visuals(routes, img_bgr, attn_root, image_name, query_index,
                       dark_ratio):
    score_dir = osp.join(attn_root, 'route_scores')
    mask_dir = osp.join(attn_root, 'top_p_mask')
    mkdir(score_dir)
    mkdir(mask_dir)
    summary = OrderedDict()

    for name, _module, route_debug, stage_idx, block_idx in routes:
        route_scores = route_debug['route_score_full_energy']
        topk_index = route_debug['topk_index']
        valid_mask = route_debug['valid_mask']
        if route_scores.dim() != 3:
            continue
        total_queries = route_scores.shape[1]
        query = min(max(0, int(query_index)), total_queries - 1)
        if query != query_index:
            print(f'query_index {query_index} exceeds {total_queries}; use {query}.')

        scores = route_scores[0, query].numpy()
        valid = valid_mask[0, query].bool().numpy()
        chosen = topk_index[0, query].numpy()[valid].astype(np.int64)
        chosen_scores = scores[chosen] if chosen.size > 0 else np.array([])
        clean_name = sanitize_name(name)
        prefix = f'{image_name}_{clean_name}_q{query}'

        score_overlay = draw_route_score_overlay(img_bgr, scores)
        cv2.imwrite(osp.join(score_dir, f'{prefix}.png'), score_overlay)

        mask_overlay = draw_top_p_mask(
            img_bgr=img_bgr,
            selected_indices=chosen,
            selected_scores=chosen_scores,
            total=scores.shape[0],
            dark_ratio=dark_ratio)
        cv2.imwrite(osp.join(mask_dir, f'{prefix}.png'), mask_overlay)

        summary[clean_name] = dict(
            stage=stage_idx,
            block=block_idx,
            query_index=query,
            route_flag=route_debug.get('route_flag'),
            configured_topk=route_debug.get('configured_topk'),
            p=route_debug.get('p'),
            temperature=route_debug.get('temperature'),
            energy=route_debug.get('energy'),
            selected_indices=chosen.tolist(),
            selected_scores=[float(x) for x in chosen_scores.tolist()])

    if summary:
        with open(osp.join(attn_root, f'{image_name}_route_summary.json'),
                  'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def save_baseline_features(feature_store, img_bgr, feature_root, reduce):
    stage_dir = osp.join(feature_root, 'fallback_stage_outputs')
    mkdir(stage_dir)
    for name, feat in feature_store.items():
        save_feature_images(
            feat=feat,
            img_bgr=img_bgr,
            save_prefix=osp.join(stage_dir, sanitize_name(name)),
            layout='nchw',
            reduce=reduce)


def save_clip_features(feature_store, text_store, img_bgr, feature_root,
                       reduce):
    stage_dir = osp.join(feature_root, 'stage_outputs')
    text_dir = osp.join(feature_root, 'text_injection')
    mkdir(stage_dir)
    mkdir(text_dir)

    injected_stages = set()
    for name, pair in text_store.items():
        stage_match = re.search(r'stages\.(\d+)\.', name)
        if stage_match:
            injected_stages.add(int(stage_match.group(1)) + 1)
        if 'before' not in pair or 'after' not in pair:
            continue
        prefix = osp.join(text_dir, name)
        save_feature_images(pair['before'], img_bgr, f'{prefix}_before_text',
                            layout='nhwc', reduce=reduce)
        save_feature_images(pair['after'], img_bgr, f'{prefix}_after_text',
                            layout='nhwc', reduce=reduce)
        save_diff_images(pair['before'], pair['after'], img_bgr,
                         f'{prefix}_text_delta', layout='nhwc')

    for name, feat in feature_store.items():
        stage_match = re.search(r'stage(\d+)_', name)
        stage_no = int(stage_match.group(1)) if stage_match else None
        if stage_no in injected_stages:
            continue
        save_feature_images(
            feat=feat,
            img_bgr=img_bgr,
            save_prefix=osp.join(stage_dir, sanitize_name(name)),
            layout='nchw',
            reduce=reduce)


def main():
    args = parse_args()
    from mmengine.config import Config
    from mmseg.apis import inference_model, init_model
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    selected_index = (
        args.test_index_opt
        if args.test_index_opt is not None else args.test_index)
    image_path = args.image or resolve_test_image_from_config(
        cfg, selected_index)
    if image_path is None:
        raise ValueError('Please specify --image or a test dataset index.')
    if not osp.isfile(image_path):
        raise FileNotFoundError(image_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f'Failed to read image: {image_path}')

    model = init_model(
        cfg,
        args.checkpoint,
        device=args.device)
    model.eval()
    if not args.keep_cuda_route:
        disable_cuda_route_backend(model)
    enable_route_debug_cache(model)

    name = image_stem(image_path)
    feature_root = osp.abspath(osp.join(args.feature_dir, args.mode, name))
    attn_root = osp.abspath(osp.join(args.attn_dir, args.mode, name))
    mkdir(feature_root)
    mkdir(attn_root)
    cv2.imwrite(osp.join(feature_root, 'original.png'), img_bgr)
    cv2.imwrite(osp.join(attn_root, 'original.png'), img_bgr)

    feature_store = OrderedDict()
    text_store = OrderedDict()
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone

    if args.mode == 'baseline':
        existing_dir = osp.join(feature_root, 'existing_logic')
        enable_existing_baseline_feature_vis(
            backbone, existing_dir, img_bgr.shape[:2])

    handles = []
    handles.extend(register_stage_hooks(backbone, feature_store))
    if args.mode == 'clip':
        handles.extend(register_text_hooks(backbone, text_store))

    try:
        inference_model(model, image_path)
    finally:
        for handle in handles:
            handle.remove()

    if args.mode == 'baseline':
        existing_dir = osp.join(feature_root, 'existing_logic')
        if not osp.isdir(existing_dir) or not os.listdir(existing_dir):
            save_baseline_features(
                feature_store, img_bgr, feature_root, args.feature_reduce)
    else:
        save_clip_features(
            feature_store, text_store, img_bgr, feature_root,
            args.feature_reduce)

    routes = collect_route_modules(model, args)
    if not routes:
        print('No Top-P route cache found. Check whether the model uses Top-P routing.')
    else:
        save_route_visuals(
            routes=routes,
            img_bgr=img_bgr,
            attn_root=attn_root,
            image_name=name,
            query_index=args.query_index,
            dark_ratio=args.dark_ratio)

    print(f'Feature maps saved to: {feature_root}')
    print(f'Attention maps saved to: {attn_root}')


if __name__ == '__main__':
    main()
