import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class YZSegDataset(BaseSegDataset):
    """YZ/GQY water-scene segmentation dataset."""

    METAINFO = dict(
        # YZ/GQY 原始标签顺序：水、地面、物体
        classes=('water', 'ground', 'object'),
        palette=[[0, 0, 255],  # water - blue
                 [0, 255, 0],  # ground - green
                 [255, 0, 0]]  # object - red
    )

    def __init__(self,
                 img_suffix='.jpg',  # 根据您的实际图像格式调整
                 seg_map_suffix='.png',  # 根据您的实际标注格式调整
                 reduce_zero_label=False,  # 重要：新数据集可能不需要减少零标签
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
