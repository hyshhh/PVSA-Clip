import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from mmengine.logging import MMLogger

from tools.analysis_tools.get_flops import inference as get_flops_inference


CNN_BLOCK_TYPES = [
    'dwconv',
    'dwconv_act',
    'mbconv',
    'mbconv_no_se',
    'c2f',
    'c3k2',
    'convnext',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CNN block ablations and summarize best mIoU.')
    parser.add_argument('config', help='Base config file path.')
    parser.add_argument(
        '--work-dir-root',
        default='ablation/vision-backbone',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
    parser.add_argument(
        '--cnn-block-types',
        nargs='+',
        choices=CNN_BLOCK_TYPES,
        default=CNN_BLOCK_TYPES,
        help='CNN block types to run.')
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
        help='Do not launch training, only refresh summary.csv from existing work_dirs.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[256, 256],
        help='Input shape used for FLOPs/Params summary.')
    return parser.parse_args()


def build_run_name(cnn_block_type):
    return f'cnn-block-{cnn_block_type}'


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


def compute_model_complexity(config_path: Path, cnn_block_type, shape):
    logger = MMLogger.get_instance(name='backbone_block_ablation_complexity')
    args = SimpleNamespace(
        config=str(config_path),
        shape=list(shape),
        cfg_options={'model.backbone.cnn_block_type': cnn_block_type},
    )
    result = get_flops_inference(args, logger)
    return result['flops'], result['params']


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cnn_block_type in args.cnn_block_types:
        run_name = build_run_name(cnn_block_type)
        work_dir = work_dir_root / run_name
        best_existing = parse_best_miou(work_dir)
        flops, params = compute_model_complexity(
            config_path, cnn_block_type, args.shape)

        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': run_name,
                'cnn_block_type': cnn_block_type,
                'flops': flops,
                'params': params,
                'best_mIoU': f'{best_existing:.6f}',
                'status': 'skipped_existing',
            })
            continue

        if args.summary_only:
            summary_rows.append({
                'run_name': run_name,
                'cnn_block_type': cnn_block_type,
                'flops': flops,
                'params': params,
                'best_mIoU': '' if best_existing is None else f'{best_existing:.6f}',
                'status': 'summary_only' if best_existing is not None else 'missing',
            })
            continue

        command = [
            args.python,
            'tools/train.py',
            str(config_path),
            '--work-dir',
            str(work_dir),
            '--cfg-options',
            f'model.backbone.cnn_block_type={cnn_block_type}',
            *args.extra_cfg_options,
        ]

        print(' '.join(command))
        if args.dry_run:
            summary_rows.append({
                'run_name': run_name,
                'cnn_block_type': cnn_block_type,
                'flops': flops,
                'params': params,
                'best_mIoU': '',
                'status': 'dry_run',
            })
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        summary_rows.append({
            'run_name': run_name,
            'cnn_block_type': cnn_block_type,
            'flops': flops,
            'params': params,
            'best_mIoU': '' if best_miou is None else f'{best_miou:.6f}',
            'status': 'ok' if result.returncode == 0 else f'failed({result.returncode})',
        })
        write_summary(work_dir_root / 'summary.csv', summary_rows)

    write_summary(work_dir_root / 'summary.csv', summary_rows)


def write_summary(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'run_name', 'cnn_block_type', 'flops', 'params',
                'best_mIoU', 'status'
            ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
