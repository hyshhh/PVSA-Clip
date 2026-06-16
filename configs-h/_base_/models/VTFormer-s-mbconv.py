_base_ = './VTFormer-s.py'

# MBConv版额外骨干：
# - 全局切换为MBConv，便于一键对比。
# - MBConv表达力比原DWConv强，因此CNN分支额外层数统一收敛到每层1个。
# - Transformer分支额外层保持0，避免路由注意力前再堆卷积导致推理变慢。
model = dict(
    backbone=dict(
        extra_block_type='mbconv',
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=1, expansion=3, kernel_size=3)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=1, expansion=3, kernel_size=3)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=1, expansion=3, kernel_size=3)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(depth=1, expansion=3, kernel_size=3)),
        ]))
