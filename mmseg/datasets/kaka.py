from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class KAKADataset(BaseSegDataset):
    """KAKA water-scene segmentation dataset."""

    METAINFO = dict(
        classes=('background', 'boat', 'free-space'),
        palette=[
            [0, 0, 0],
            [128, 0, 128],
            [0, 0, 255],
        ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=255,
            **kwargs)
