# Copyright (c) OpenMMLab. All rights reserved.
"""让 `clip_embed_dim` 成为真正可控的可选参数。

背景：base 模型文件 `clip-topp.py` 通过 `exec` 注入，其中用
``globals().get('clip_embed_dim', 512)`` 在 ``exec`` 阶段就把 512
固化进 ``cfg.model.decode_head.embed_dim`` 等三处。而
``--cfg-options clip_embed_dim=256`` 走的是 ``Config.merge_from_dict``，
发生在 ``exec`` 之后、且只改顶层标量键，无法回头重跑 ``exec``，
因此模型结构里的维度仍是 512，命令实际失效。

修复：在 merge 完成、构建模型之前调用本模块的 :func:`sync_clip_embed_dim`，
若顶层存在 ``clip_embed_dim``，就把它的值同步回写到 ``cfg.model``
中被固化的三处位置，使可选维度真正生效。
"""


def sync_clip_embed_dim(cfg):
    """把顶层 ``clip_embed_dim`` 同步回写到模型结构中被固化的维度键。

    仅当顶层 ``clip_embed_dim`` 存在时才动作，且只为已存在的对应字段赋值，
    不新增、不改动其它配置；首层仅支持 dict 形式的 decode_head /
    text_encoder / text_refiner。

    Args:
        cfg (Config): 已经 merge 过 ``cfg_options`` 的配置对象。
    """
    clip_embed_dim = cfg.get('clip_embed_dim', None)
    if clip_embed_dim is None:
        return

    model = cfg.get('model', None)
    if not isinstance(model, dict):
        return

    # decode_head / text_encoder 期望为 dict；text_refiner 可能为 None 或 dict。
    for section_name, key in (('decode_head', 'embed_dim'),
                              ('text_encoder', 'embed_dim')):
        section = model.get(section_name, None)
        if isinstance(section, dict) and key in section:
            section[key] = clip_embed_dim

    text_refiner = model.get('text_refiner', None)
    if isinstance(text_refiner, dict) and 'in_dim' in text_refiner:
        text_refiner['in_dim'] = clip_embed_dim
