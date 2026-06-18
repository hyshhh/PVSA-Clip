_base_ = './VTFormer-s.py'

# MBConv版额外骨干：
# - 全局切换为MBConv，便于一键对比。
# - 加入SE注意力，使用更标准的MBConv结构。
# - layer_scale从1e-6调到1e-5，让新块更早参与训练但仍保持稳定。
# - Transformer分支额外层保持0，避免路由注意力前再堆卷积导致推理变慢。
model = dict(
    backbone=dict(
        extra_block_type='mbconv',
        mask_fusion_scale=0.5,
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=3, kernel_size=3,
                    se_ratio=0.25, layer_scale=1e-5)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=3, kernel_size=3,
                    se_ratio=0.25, layer_scale=1e-5)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=3, kernel_size=3,
                    se_ratio=0.25, layer_scale=1e-5)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=3, kernel_size=3,
                    se_ratio=0.25, layer_scale=1e-5)),
        ]))
