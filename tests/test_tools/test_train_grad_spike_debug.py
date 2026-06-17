from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[2]


def test_train_script_enables_grad_spike_debug_after_cfg_merge():
    source = (_repo_root() / 'tools' / 'train.py').read_text(encoding='utf-8')
    config = (
        _repo_root()
        / 'configs-h'
        / 'biformer'
        / 'biformer_mm-20k_chase_db1-512x512.py'
    ).read_text(encoding='utf-8')

    assert "if cfg.get('grad_spike_debug', False):" in source
    assert "cfg.custom_hooks = custom_hooks" in source
    assert 'grad_spike_debug = False' in config
    assert 'if grad_spike_debug:' not in config
