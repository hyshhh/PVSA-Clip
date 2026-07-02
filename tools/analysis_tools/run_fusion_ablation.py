import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


FUSION_TYPES = [
    'conv1x1',
    'conv1x1_bn_gelu',
    'conv1x1_bn_gelu_dwconv',
]

CROSS_STAGE_FUSION_MODES = [
    'none',
    'gate',
    'concat',
    'gate_concat',
    'cross_gate',
    'cross_concat',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run fusion ablation experiments and summarize best mIoU.')
    parser.add_argument('config', help='Base config file path.')
    parser.add_argument(
        '--work-dir-root',
        default='work_dirs/fusion_ablation',
        help='Root directory for ablation runs.')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python executable used to launch training.')
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
    return parser.parse_args()


def build_run_name(fusion_type, cross_stage_mode):
    return f'fusion-{fusion_type}__cross-{cross_stage_mode}'


def parse_best_miou(work_dir: Path):
    best_ckpt = work_dir / 'best_mIoU.pth'
    if best_ckpt.exists():
        timestamp_files = sorted(work_dir.glob('*.json'))
        for json_file in reversed(timestamp_files):
            best = parse_best_miou_from_json(json_file)
            if best is not None:
                return best

    for json_file in sorted(work_dir.glob('*.json')):
        best = parse_best_miou_from_json(json_file)
        if best is not None:
            return best
    return None


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
                if 'mIoU' in record and isinstance(record['mIoU'], (int, float)):
                    best = max(best, float(record['mIoU'])) if best is not None else float(record['mIoU'])
    except OSError:
        return None
    return best


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    work_dir_root = Path(args.work_dir_root).resolve()
    work_dir_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for fusion_type, cross_stage_mode in itertools.product(
            FUSION_TYPES, CROSS_STAGE_FUSION_MODES):
        run_name = build_run_name(fusion_type, cross_stage_mode)
        work_dir = work_dir_root / run_name
        best_existing = parse_best_miou(work_dir)
        if args.skip_existing and best_existing is not None:
            summary_rows.append({
                'run_name': run_name,
                'fusion_type': fusion_type,
                'cross_stage_fusion_mode': cross_stage_mode,
                'best_mIoU': best_existing,
                'status': 'skipped_existing',
            })
            continue

        command = [
            args.python,
            'tools/train.py',
            str(config_path),
            '--work-dir',
            str(work_dir),
            '--cfg-options',
            f'model.backbone.fusion_type={fusion_type}',
            f'model.backbone.cross_stage_fusion_mode={cross_stage_mode}',
            *args.extra_cfg_options,
        ]

        print(' '.join(command))
        if args.dry_run:
            summary_rows.append({
                'run_name': run_name,
                'fusion_type': fusion_type,
                'cross_stage_fusion_mode': cross_stage_mode,
                'best_mIoU': '',
                'status': 'dry_run',
            })
            continue

        result = subprocess.run(command, check=False)
        best_miou = parse_best_miou(work_dir)
        summary_rows.append({
            'run_name': run_name,
            'fusion_type': fusion_type,
            'cross_stage_fusion_mode': cross_stage_mode,
            'best_mIoU': '' if best_miou is None else f'{best_miou:.6f}',
            'status': 'ok' if result.returncode == 0 else f'failed({result.returncode})',
        })

        summary_path = work_dir_root / 'summary.csv'
        write_summary(summary_path, summary_rows)

    write_summary(work_dir_root / 'summary.csv', summary_rows)


def write_summary(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['run_name', 'fusion_type', 'cross_stage_fusion_mode', 'best_mIoU', 'status'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
