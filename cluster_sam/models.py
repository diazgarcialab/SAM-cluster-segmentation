# from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pycocotools import mask as cocomask  # type: ignore

import numpy as np
import numpy.typing as npt
import cv2  # type: ignore
import pandas as pd
from plotly.graph_objects import Figure, Scatter
import plotly.express as px


class ModelType(str, Enum):
    DEFAULT = "default"
    VIT_H = "vit_h"
    VIT_L = "vit_l"
    VIT_B = "vit_b"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class MaskFormat(str, Enum):
    UNCOMPRESSED_RLE = "uncompressed_rle"
    COCO_RLE = "coco_rle"
    BINARY_MASK = "binary_mask"


# OpenCV type hint for image
CV2Image = cv2.typing.MatLike


@dataclass
class AMGSettings:
    points_per_side: Optional[int] = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: float = 1
    point_grids: Optional[List[np.ndarray]] = None
    min_mask_region_area: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


@dataclass
class ModelMeta:
    name: str
    version: str
    checkpoint: str
    device: str
    mask_format: str
    model_settings: AMGSettings


@dataclass
class BBox:
    x_min: float
    y_min: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x_min + self.width

    @property
    def y_max(self) -> float:
        return self.y_min + self.height

    @property
    def center(self) -> Tuple[float, float]:
        return self.x_min + self.width / 2, self.y_min + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        x, y = point
        return (
            self.x_min <= x <= self.x_min + self.width
            and self.y_min <= y <= self.y_min + self.height
        )

    def to_yolo(
        self, img_width: int, img_height: int, round_to: int = 6
    ) -> Tuple[float, float, float, float]:
        x_center = round((self.x_min + self.width / 2) / img_width, round_to)
        y_center = round((self.y_min + self.height / 2) / img_height, round_to)
        w = round(self.width / img_width, round_to)
        h = round(self.height / img_height, round_to)
        return x_center, y_center, w, h

    @classmethod
    def from_list(cls, l: List[float]) -> "BBox":
        return BBox(x_min=l[0], y_min=l[1], width=l[2], height=l[3])

    def plot(
        self,
        name: str = "BBox",
        fig: Figure = None,
        color: str = None,
        mode: str = "lines",
        fill: bool = False,
    ) -> Figure:
        if fig is None:
            fig = Figure()
            fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
        fig.add_trace(
            Scatter(
                x=[self.x_min, self.x_max, self.x_max, self.x_min, self.x_min],
                y=[self.y_min, self.y_min, self.y_max, self.y_max, self.y_min],
                mode=mode,
                name=name,
                showlegend=True,
                fill="toself" if fill else "none",
                opacity=0.9,
                hoverinfo="x+y+name",
                line=dict(color=color, width=2) if color is not None else dict(width=2),
                hoverlabel=dict(bgcolor="white", font_size=16, namelength=-1),
            )
        )
        return fig


@dataclass
class Segmentation:
    size: List[int]
    counts: List[int] | str
    # Cache
    _contour: np.ndarray = None
    _binary_mask: np.ndarray = None
    _centroid: np.ndarray = None

    @property
    def contour(
        self,
        remove_small_objects: bool = False,
        min_area: int = 200,
        merge_contours=False,
    ) -> npt.ArrayLike:
        if self._contour is None:
            mask_decoded = self.decode()
            contours, hierarchy = cv2.findContours(
                mask_decoded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if remove_small_objects:
                contours = [c for c in contours if cv2.contourArea(c) > min_area]
            if merge_contours:
                contour = np.concatenate(contours)
            else:
                contour = contours[0]
            self._contour = contour
        return self._contour

    @property
    def area(self) -> float:
        return int(cv2.contourArea(self.contour))

    @property
    def binary_mask(self) -> npt.NDArray[np.uint8]:
        if self._binary_mask is None:
            self._binary_mask = self.decode()
        return self._binary_mask

    @property
    def centroid(self) -> np.ndarray:
        if self._centroid is None:
            cnt = self.contour
            M = cv2.moments(cnt)
            self._centroid = np.array(
                [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
            )
        return self._centroid

    @property
    def rle(self) -> Dict[str, Any]:
        return dict(size=self.size, counts=self.counts)

    @property
    def bbox(self) -> BBox:
        return BBox.from_list(cocomask.toBbox(self.rle).ravel())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Segmentation":
        return Segmentation(size=d["size"], counts=d["counts"])

    # Add two Segmentation objects
    def __add__(self, other):
        return Segmentation.from_dict(cocomask.merge([self.to_dict(), other.to_dict()]))

    def to_dict(self) -> Dict[str, Any]:
        return self.rle

    def decode(self) -> np.ndarray:
        d = {"size": self.size, "counts": self.counts}
        return cocomask.decode(d)

    def rle_uncompressed(self) -> Dict[str, Any]:
        # https://www.kaggle.com/code/stainsby/fast-tested-rle/notebook
        pixels = self.binary_mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return {"size": self.size, "counts": runs.tolist()}

    def plot(
        self,
        name: str = "Contour",
        text: str = "Contour",
        fig: Figure = None,
        color: str = None,
        mode: str = "lines",
        fill: bool = True,
        plot_bbox: bool = False,
        plot_centroid: bool = False,
    ) -> Figure:
        if fig is None:
            fig = Figure()
            fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
        contour = self.contour.squeeze()
        line_style = dict(color=color, width=2) if color is not None else dict(width=2)
        fig.add_trace(
            Scatter(
                x=contour[:, 0],
                y=contour[:, 1],
                mode=mode,
                text=text,
                name=name,
                showlegend=True,
                fill="toself" if fill else "none",
                opacity=0.9,
                hoverinfo="x+y+text+name",
                line=line_style,
                hoverlabel=dict(
                    bgcolor="rgba(255,255,255,0.7)", font_size=16, namelength=-1
                ),
            )
        )
        if plot_bbox:
            fig = self.bbox.plot(fig=fig, color="red", mode="lines", fill=False)
        if plot_centroid:
            fig.add_trace(
                Scatter(
                    x=[self.centroid[0]],
                    y=[self.centroid[1]],
                    mode="markers+text",
                    name="Centroid",
                    marker=dict(
                        size=8,
                        symbol="star",
                        color="orange",
                        line=dict(width=1, color="black"),
                    ),
                    text="<b>Centroid</b>",
                    textposition="bottom center",
                    textfont=dict(color="black", size=8),
                    hoverinfo="x+y+name",
                )
            )
        return fig

    def plot_binary(
        self, color_continuous_scale: str = "purples", plot_bbox: bool = True
    ) -> Figure:
        fig = px.imshow(
            self.binary_mask,
            color_continuous_scale=color_continuous_scale,
            labels=dict(x="X", y="Y"),
        )
        if plot_bbox:
            self.bbox.plot(fig=fig, color="red", mode="lines", fill=False)
        fig.update_layout(
            yaxis=dict(autorange="reversed", scaleratio=1), coloraxis_showscale=False
        )
        return fig


@dataclass
class Mask:
    segmentation: Segmentation
    area: float
    bbox: List[float]
    predicted_iou: float
    point_coords: List[List[float]]
    stability_score: float
    crop_box: List[float]
    iou: Optional[float] = None
    label: Optional[str] = None
    area_normalized: Optional[float] = None

    def to_dict(self, flat=False) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        if flat:
            seg = d.pop("segmentation")
            d["segmentation_counts"] = seg["counts"]
            d["segmentation_size"] = seg["size"]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        # Check required fields
        required_fields = [
            "segmentation_size",
            "segmentation_counts",
            "area",
            "bbox",
            "predicted_iou",
            "point_coords",
            "stability_score",
            "crop_box",
        ]
        optional_fields = ["iou", "label", "area_normalized"]
        for field in required_fields:
            assert field in d, f"Missing field {field}"
        # Catch required fields
        d_clean = {k: v for k, v in d.items() if k in required_fields}
        # Catch optional fields
        for k in optional_fields:
            if k in d:
                d_clean[k] = d[k]
        d_clean["segmentation"] = Segmentation(
            size=d_clean.pop("segmentation_size"),
            counts=d_clean.pop("segmentation_counts"),
        )
        return Mask(**d_clean)

    def plot(
        self,
        name: str,
        fig: Figure = None,
        color: str = None,
        mode: str = "lines",
        fill: bool = True,
    ) -> Figure:
        return self.segmentation.plot(
            name=name, fig=fig, color=color, mode=mode, fill=fill
        )


@dataclass
class SegmentationResult:
    created_date: datetime
    file_name: str
    roi_bbox: BBox
    model_metadata: ModelMeta
    masks: List[Mask]

    @classmethod
    def from_json(cls, json_file: Path):
        with open(json_file, "r", encoding="utf-8") as f:
            s = SegmentationResult(**json.load(f))
            s.roi_bbox = BBox(**s.roi_bbox)
            s.model_metadata = ModelMeta(**s.model_metadata)
            s.model_metadata.model_settings = AMGSettings(
                **s.model_metadata.model_settings
            )
            s.masks = [Mask(**m) for m in s.masks]
            for mask in s.masks:
                mask.segmentation = Segmentation(**mask.segmentation)
        return s


@dataclass
class SAMMasksInFace:
    plant_id: str
    cluster_id: int
    face_id: str
    original_image_path: str
    roi_box: BBox
    masks: List[Mask]

    def to_list(self, flat=False):
        data = []
        for mask in self.masks:
            d = {
                "plant_id": self.plant_id,
                "cluster_id": self.cluster_id,
                "face_id": self.face_id,
                "original_image_path": self.original_image_path,
                "roi_x_min": self.roi_box.x_min,
                "roi_y_min": self.roi_box.y_min,
                "roi_width": self.roi_box.width,
                "roi_height": self.roi_box.height,
                **mask.to_dict(flat=flat),
            }
            data.append(d)
        return data

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "SAMMasksInFace":
        # Check required columns
        required_cols = [
            "plant_id",
            "cluster_id",
            "face_id",
            "original_image_path",
            "roi_x_min",
            "roi_y_min",
            "roi_width",
            "roi_height",
            "area",
            "bbox",
            "predicted_iou",
            "point_coords",
            "stability_score",
            "crop_box",
            "segmentation_size",
            "segmentation_counts",
        ]
        optional_cols = ["iou", "label", "area_normalized"]
        for col in required_cols:
            assert col in df.columns, f"Column {col} not found in dataframe"

        d = {k: v for k, v in df.iloc[0].to_dict().items() if k in required_cols}

        sam_masks_in_face = SAMMasksInFace(
            plant_id=d.pop("plant_id"),
            cluster_id=d.pop("cluster_id"),
            face_id=d.pop("face_id"),
            original_image_path=d.pop("original_image_path"),
            roi_box=BBox(
                x_min=d.pop("roi_x_min"),
                y_min=d.pop("roi_y_min"),
                width=d.pop("roi_width"),
                height=d.pop("roi_height"),
            ),
            masks=[],
        )
        # Load masks
        for _, row in df.iterrows():
            mask = Mask.from_dict(row.to_dict())
            sam_masks_in_face.masks.append(mask)

        return sam_masks_in_face

    def to_df(self, flat=True) -> pd.DataFrame:
        data = self.to_list(flat=flat)
        return pd.DataFrame(data)
