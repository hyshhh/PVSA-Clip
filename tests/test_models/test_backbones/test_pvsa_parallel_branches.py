from pathlib import Path


def test_biformer_fusion_has_cuda_stream_parallel_branch_path():
    root = Path(__file__).resolve().parents[3]
    source = (
        root / 'mmseg' / 'models' / 'backbones' / 'biformer_fusion.py'
    ).read_text(encoding='utf-8')

    assert 'def run_parallel_branches' in source
    assert 'torch.cuda.Stream' in source
    assert 'trans_stream.wait_stream(cnn_stream)' in source
    assert 'next_cnn.record_stream(trans_stream)' in source
