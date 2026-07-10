from typing import Optional, Tuple


def split_backbone_text_inputs(category_prototypes) -> Tuple[Optional[object], Optional[object]]:
    """统一骨干文本输入格式。

    支持三种输入：
    1. None
    2. 单个 Tensor，默认同时作为 route_text / align_text
    3. dict(route_text=..., align_text=...)
    """
    if category_prototypes is None:
        return None, None
    if isinstance(category_prototypes, dict):
        route_text = category_prototypes.get('route_text', None)
        align_text = category_prototypes.get('align_text', None)
        if route_text is None:
            route_text = align_text
        if align_text is None:
            align_text = route_text
        return route_text, align_text
    return category_prototypes, category_prototypes
