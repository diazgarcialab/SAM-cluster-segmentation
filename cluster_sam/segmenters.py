from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2  # type: ignore
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from models import (
    ModelMeta,
    Mask,
    MaskFormat,
    ModelType,
    Device,
    BBox,
    AMGSettings,
    SegmentationResult,
    CV2Image,
)


class Segmenter(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ModelMeta:
        raise NotImplementedError()

    @abstractmethod
    def generate(self, img: CV2Image) -> List[Mask]:
        raise NotImplementedError()


class SAMSegmenter(Segmenter):
    __NAME__ = "SAMG"
    __VERSION__ = "1.0"

    def __init__(
        self,
        model_type: ModelType,
        checkpoint: Path,
        device: Device,
        mask_format: MaskFormat,
        amg_settings: AMGSettings,
    ):
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
        self.mask_format = mask_format
        self.amg_settings = amg_settings
        self.generator = self.init_generator()

    @property
    def metadata(self) -> ModelMeta:
        return ModelMeta(
            name=self.__NAME__,
            version=self.__VERSION__,
            checkpoint=str(self.checkpoint),
            device=self.device,
            mask_format=self.mask_format,
            model_settings=self.amg_settings,
        )

    def init_generator(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        _ = sam.to(device=self.device)
        return SamAutomaticMaskGenerator(
            sam, output_mode=self.mask_format, **self.amg_settings.to_dict()
        )

    def generate(self, img: CV2Image) -> List[Mask]:
        masks_raw = self.generator.generate(img)
        return [Mask(**mask) for mask in masks_raw]


class ClusterSegmenter:
    def __init__(self, segmenter: Segmenter):
        self.mask_generator = segmenter

    def load_image(self, file_name: Path) -> CV2Image:
        img = cv2.imread(str(file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def crop_cluster(
        self,
        img_orig: CV2Image,
        bbox: BBox = None,
        x_offset: int = 100,
        y_offset: int = 100,
    ) -> Tuple[CV2Image, Sequence[cv2.typing.MatLike] | None, BBox]:
        img = img_orig.copy()

        if bbox is not None:
            contour = None
            x_min, x_max = max(0, bbox.x_min - x_offset), min(
                img.shape[1], bbox.x_min + bbox.width + x_offset
            )
            y_min, y_max = max(0, bbox.y_min - y_offset), min(
                img.shape[0], bbox.y_min + bbox.height + y_offset
            )
        else:
            # Try to find the cluster if no bbox is provided
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            (T, thresh) = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Choose the largest contour
            contour = contours[0]  # TODO This is not robust enough
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            x_min, x_max = max(0, x - x_offset), min(img.shape[1], x + w + x_offset)
            y_min, y_max = max(0, y - y_offset), min(img.shape[0], y + h + y_offset)

        bbox = BBox(x_min=x_min, y_min=y_min, width=x_max - x_min, height=y_max - y_min)
        return img[y_min:y_max, x_min:x_max], contour, bbox

    def generate(
        self, file_name: Path, bbox: BBox = None, dry_run=False
    ) -> SegmentationResult:
        img = self.load_image(file_name)
        img_cluster, contour, bbox = self.crop_cluster(img, bbox=bbox)
        if dry_run:
            masks = []
        else:
            masks = self.mask_generator.generate(img_cluster)
        return SegmentationResult(
            created_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            file_name=str(file_name),
            roi_bbox=bbox,
            model_metadata=self.mask_generator.metadata,
            masks=masks,
        )
