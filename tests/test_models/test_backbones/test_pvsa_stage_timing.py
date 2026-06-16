from pathlib import Path


def test_biformer_fusion_reports_parallel_wall_time_only():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'biformer_fusion.py'
    ).read_text(encoding='utf-8')

    assert 'def _time_cuda_wall' in source
    assert 'stage_total_wall' in source
    assert 'PVSA_SERIAL_STAGE_PROFILE' not in source
    assert 'serial_stage_profile' not in source
    assert 'serial_cnn_branch' not in source
    assert 'serial_trans_down' not in source
    assert 'serial_trans_stage' not in source
    assert "'cnn_branch'" not in source
    assert "'trans_down'" not in source
    assert "'trans_stage'" not in source
