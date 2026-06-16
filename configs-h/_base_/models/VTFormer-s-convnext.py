_base_ = './VTFormer-s.py'

# ConvNeXt版额外骨干：
# - 全局切换为ConvNeXt块，增强局部建模。
# - ConvNeXt在高分辨率stage0较吃算力，所以stage0只放1个轻量块。
# - stage3空间尺寸小，保留1个块提升语义层表达，整体是精度优先的中等容量配置。
model = dict(
    backbone=dict(
        extra_block_type='convnext',
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=2, kernel_size=7,
                    layer_scale=1e-6)),
        ]))
