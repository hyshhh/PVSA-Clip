from pathlib import Path


def test_vtformer_exposes_cnn_branch_depth_config():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'bi_topp_vote.py'
    ).read_text(encoding='utf-8')
    config = (
        root / 'configs-h' / '_base_' / 'models' / 'VTFormer-s.py'
    ).read_text(encoding='utf-8')

    assert 'cnn_branch_depth=None' in source
    assert 'self.cnn_branch_depth[0]' in source
    assert 'self.cnn_branch_depth[i + 1]' in source
    assert 'cnn_branch_depth=[2, 1, 2, 1]' in config
