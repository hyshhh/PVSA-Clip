_base_ = './VTFormer-s.py'

# ConvNeXt版额外骨干：
# - 全局切换为ConvNeXt块，增强局部建模。
# - 深度保持和原始DWConv版一致，只切换块类型，方便公平对比。
# - 使用适中的layer_scale，让新增块更早参与训练，同时保持残差稳定。
model = dict(
    backbone=dict(
        extra_block_type='convnext',
        mask_fusion_scale=0.5,
        stage_archs=[
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=4, kernel_size=7,
                    layer_scale=1e-4)),
            dict(
                blocks=4,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=4, kernel_size=7,
                    layer_scale=1e-4)),
            dict(
                blocks=6,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=2, expansion=4, kernel_size=7,
                    layer_scale=1e-4)),
            dict(
                blocks=3,
                trans_extra=dict(depth=0),
                cnn_extra=dict(
                    depth=1, expansion=4, kernel_size=7,
                    layer_scale=1e-4)),
        ]))
