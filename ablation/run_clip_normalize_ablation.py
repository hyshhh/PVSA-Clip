
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from mmengine.logging import MMLogger

from tools.analysis_tools.get_flops import inference as get_flops_inference



NORMALIZE_VARIANTS = [
    'norm-N0-topp-dot',  # ToppAttention + 默认点积
    'norm-N1-topp-cos',  # ToppAttention + 严格余弦
    'norm-N2-brg-dot',   # BiFormer + 默认点积
    'norm-N3-brg-cos',   # BiFormer + 严格余弦
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CLIP normalize-visual ablations and summarize '
        'best mIoU.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/clip-normalize',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
    parser.add_argument(
        '--variants',
        nargs='+',
        choices=NORMALIZE_VARIANTS,
        default=NORMALIZE_VARIANTS,
        help='Variant names to run (subset of the four N0-N3 groups).')
    parser.add_argument(
        '--extra-cfg-options',
        nargs='*',
        default=[],
        help='Additional cfg-options passed through to tools/train.py.')
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip runs that already have a best mIoU result.')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print commands without launching training.')
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Do not launch training, only refresh summary.csv from existing '
             'work_dirs.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Input shape used for FLOPs/Params summary.')
    return parser.parse_args()


def get_variant(variant_name):
    """按变体名查表返回 (base_config_rel, normalize_visual)。

    对照矩阵：
        N0  ToppAttention + 默认点积   base=waterseg.py       normalize_visual=False
        N1  ToppAttention + 严格余弦   base=waterseg.py       normalize_visual=True
        N2  BiFormer        + 默认点积   base=attn_waterseg.py  normalize_visual=False
        N3  BiFormer        + 严格余弦   base=attn_waterseg.py  normalize_visual=True
    base_config 相对仓库根；normalize_visual 通过 --cfg-options 覆盖。
    """
    if variant_name == 'norm-N0-topp-dot':
        return 'configs-h/clip/waterseg.py', False
    if variant_name == 'norm-N1-topp-cos':
        return 'configs-h/clip/waterseg.py', True
    if variant_name == 'norm-N2-brg-dot':
        return 'configs-h/clip/attn_waterseg.py', False
    if variant_name == 'norm-N3-brg-cos':
        return 'configs-h/clip/attn_waterseg.py', True
    raise ValueError(f'Unknown variant: {variant_name}. '
                     f'Valid: {NORMALIZE_VARIANTS}')


def build_cfg_options(normalize_visual):
    """返回 dict 与 list 两种形式：dict 给 get_flops，list 给 --cfg-options。"""
    cfg_dict = {'model.decode_head.normalize_visual': normalize_visual}
    cfg_list = [
        f'model.decode_head.normalize_visual='
        f'{"True" if normalize_visual else "False"}'
    ]
    return cfg_dict, cfg_list


def parse_best_miou(work_dir: Path):
    if not work_dir.exists():
        return None

    candidates = []
    candidates.extend(sorted(work_dir.rglob('scalars.json')))
    candidates.extend(sorted(
        json_file for json_file in work_dir.rglob('*.json')
        if json_file.name != 'summary.csv' and json_file.name != 'scalars.json'))

    best = None
    for json_file in candidates:
        value = parse_best_miou_from_json(json_file)
        if value is not None:
            best = value if best is None else max(best, value)
    return best


def parse_best_miou_from_json(json_file: Path):
    best = None
    try:
        with json_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key, value in record.items():
                    if 'mIoU' not in key or not isinstance(value, (int, float)):
                        continue
                    value = float(value)
                    best = value if best is None else max(best, value)
    except OSError:
        return None
    return best


def compute_model_complexity(config_path: Path, cfg_options: dict, shape):
    logger = MMLogger.get_instance(name='clip_normalize_ablation_complexity')
    args = SimpleNamespace(
        config=str(config_path),
        shape=list(shape),
        cfg_options=cfg_options,
    )
    result = get_flops_inference(args, logger)
    return result['flops'], result['params']


def main():
    args = parse_args()
    # 脚本位于 ablation/ 下，仓库根 = parents[1]（原在 tools/analysis_tools/
    # 时是 parents[2]）；变体里的 base_config 是相对仓库根的路径。
    repo_root = Path(__file__).resolve().parents[1]
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for variant_name in args.variants:
        base_rel, normalize_visual = get_variant(variant_name)
        config_path = (repo_root / base_rel).resolve()
        work_dir = work_dir_root / variant_name
        best_existing = parse_best_miou(work_dir)
        cfg_dict, cfg_list = build_cfg_options(normalize_visual)
        flops, params = compute_model_complexity(
            config_path, cfg_dict, args.shape)

        cfg_opts_str = f'normalize_visual={normalize_visual}'

        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_rel,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': f'{best_existing:.6f}',
                'status': 'skipped_existing',
            })
            continue

        if args.summary_only:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_rel,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': '' if best_existing is None
                else f'{best_existing:.6f}',
                'status': 'summary_only' if best_existing is not None
                else 'missing',
            })
            continue

        command = [
            args.python,
            'tools/train.py',
            str(config_path),
            '--work-dir',
            str(work_dir),
            '--cfg-options',
            *cfg_list,
            *args.extra_cfg_options,
        ]

        print(' '.join(command))
        if args.dry_run:
            summary_rows.append({
                'run_name': variant_name,
                'base_config': base_rel,
                'cfg_options': cfg_opts_str,
                'flops': flops,
                'params': params,
                'best_mIoU': '',
                'status': 'dry_run',
            })
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        summary_rows.append({
            'run_name': variant_name,
            'base_config': base_rel,
            'cfg_options': cfg_opts_str,
            'flops': flops,
            'params': params,
            'best_mIoU': '' if best_miou is None
            else f'{best_miou:.6f}',
            'status':
            'ok' if result.returncode == 0 else f'failed({result.returncode})',
        })
        write_summary(work_dir_root / 'summary.csv', summary_rows)

    write_summary(work_dir_root / 'summary.csv', summary_rows)


def write_summary(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'run_name', 'base_config', 'cfg_options', 'flops', 'params',
                'best_mIoU', 'status'
            ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
