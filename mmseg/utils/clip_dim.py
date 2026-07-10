# Copyright (c) OpenMMLab. All rights reserved.
"""同步头部与骨干的文本维度配置。

背景：base 模型文件会在 ``exec`` 阶段把维度直接固化进 ``cfg.model``。
而 ``--cfg-options`` 发生在其后，只会改顶层标量，无法自动回写已经固化
到 ``model`` 内部的字段。

当前约定：
- ``clip_embed_dim`` 为旧别名，仅用于头部文本宽度兼容
- ``head_clip_dim`` 控制解码头/头部文本编码宽度
- ``backbone_text_dim`` 控制骨干文本注入宽度

因此在 merge 完成、构建模型之前，需要把这些顶层值同步回写到
``cfg.model`` 中对应的嵌套字段。
"""


def sync_clip_embed_dim(cfg):
    """把顶层文本维度同步回写到模型结构中已固化的字段。

    同步规则：
    - ``head_clip_dim`` 优先；若不存在则回退到 ``clip_embed_dim``
    - ``backbone_text_dim`` 仅回写骨干文本注入链路

    Args:
        cfg (Config): 已 merge 完成的配置对象。
    """
    head_clip_dim = cfg.get('head_clip_dim', cfg.get('clip_embed_dim', None))
    backbone_text_dim = cfg.get('backbone_text_dim', None)

    model = cfg.get('model', None)
    if not isinstance(model, dict):
        return

    if head_clip_dim is not None:
        for section_name, key in (('decode_head', 'embed_dim'),
                                  ('text_encoder', 'embed_dim')):
            section = model.get(section_name, None)
            if isinstance(section, dict) and key in section:
                section[key] = head_clip_dim

    backbone_text_encoder = model.get('backbone_text_encoder', None)
    if isinstance(backbone_text_encoder, dict) and backbone_text_dim is not None:
        if 'embed_dim' in backbone_text_encoder:
            backbone_text_encoder['embed_dim'] = backbone_text_dim

    backbone = model.get('backbone', None)
    if isinstance(backbone, dict) and backbone_text_dim is not None and 'text_dim' in backbone:
        backbone['text_dim'] = backbone_text_dim

    text_refiner = model.get('text_refiner', None)
    if isinstance(text_refiner, dict) and backbone_text_dim is not None:
        if 'in_dim' in text_refiner:
            text_refiner['in_dim'] = backbone_text_dim
