from pathlib import Path


def test_biformer_fusion_reports_wall_time_and_serial_diagnostics():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'biformer_fusion.py'
    ).read_text(encoding='utf-8')

    assert 'def _time_cuda_wall' in source
    assert 'stage_total_wall' in source
    assert 'serial_cnn_branch' in source
    assert 'serial_trans_down' in source
    assert 'serial_trans_stage' in source
    assert "'cnn_branch'" not in source
    assert "'trans_down'" not in source
    assert "'trans_stage'" not in source
