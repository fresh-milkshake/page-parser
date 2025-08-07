from typing import List

from doclayout_yolo import YOLOv10
from doclayout_yolo.engine.results import Results
import numpy as np

from src.common.logging import get_logger
from src.config.settings import Settings

from .base import BaseDetectionBackend, DetectionResult


logger = get_logger(__name__)


class DoclayoutYoloBackend(BaseDetectionBackend):
    """
    Backend for document layout detection using YOLOv10 from doclayout_yolo.

    Attributes:
        model (YOLOv10): The loaded YOLOv10 model for detection.
    """

    def __init__(self, model_path: str, settings: Settings) -> None:
        """
        Initialize the DoclayoutYoloBackend with a YOLOv10 model.

        Args:
            model_path (str): Path to the YOLOv10 model file.
        """
        logger.info(f"Initializing DoclayoutYoloBackend with model: {model_path}")
        self.model = YOLOv10(model_path)
        logger.info("DoclayoutYoloBackend initialized successfully")

        self.imgsz = settings.layout_detection.doclayout_yolo.imgsz
        self.conf = settings.layout_detection.doclayout_yolo.conf
        self.device = settings.layout_detection.doclayout_yolo.device

    def detect(
        self,
        image: np.ndarray,
    ) -> List[DetectionResult]:
        """
        Run document layout detection on an image.

        Args:
            image (np.ndarray): Image to detect objects in.
            imgsz (int, optional): Prediction image size. Defaults to 1024.
            conf (float, optional): Confidence threshold. Defaults to 0.7.
            device (str, optional): Device to use for inference (e.g., 'cuda:0' or 'cpu'). Defaults to "cpu".

        Returns:
            List[DetectionResult]: List of detection results.
        """
        logger.debug(f"Running layout detection on shape: {image.shape}")

        results: List[Results] = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
        )

        detections: List[DetectionResult] = []
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes
                height, width = result.orig_shape

            if not (
                hasattr(boxes, "cls")
                and hasattr(boxes, "conf")
                and hasattr(boxes, "xyxyn")
            ):
                continue

            for label, box, conf in zip(
                boxes.cls.tolist(),
                boxes.xyxyn.tolist(),
                boxes.conf.tolist(),
            ):
                label_idx = int(label)
                bbox = [
                    float(box[0]) * width,
                    float(box[1]) * height,
                    float(box[2]) * width,
                    float(box[3]) * height,
                ]
                confidence = float(conf)
                detection = DetectionResult(
                    label=label_idx,
                    label_name=result.names[label_idx],
                    bbox=bbox,
                    confidence=confidence,
                )
                detections.append(detection)
                logger.debug(
                    f"Detection: {detection.label_name} at {bbox} (conf: {confidence:.3f})"
                )

        logger.info(f"Layout detection completed: {len(detections)} objects found")

        count_before = len(detections)
        detections = self.remove_duplicates(detections)
        count_after = len(detections)
        logger.info(f"Removed {count_before - count_after} duplicates")

        return detections

    def remove_duplicates(
        self, detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        """
        Remove duplicate or overlapping detections using Intersection over Union (IoU).
        Keeps the detection with the highest confidence for each overlapping group.

        Args:
            detections (list[DetectionResult]): List of detection results.

        Returns:
            list[DetectionResult]: List of filtered detection results.
        """

        def iou(box1: list[float], box2: list[float]) -> float:
            """Compute Intersection over Union (IoU) of two bounding boxes."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
            area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
            union_area = area1 + area2 - inter_area
            if union_area == 0.0:
                return 0.0
            return inter_area / union_area

        filtered: list[DetectionResult] = []
        detections_sorted = sorted(detections, key=lambda d: d.confidence, reverse=True)
        used = [False] * len(detections_sorted)
        iou_threshold = 0.5

        for i, det in enumerate(detections_sorted):
            if used[i]:
                continue
            filtered.append(det)
            for j in range(i + 1, len(detections_sorted)):
                if used[j]:
                    continue
                # Only suppress if same label
                if det.label == detections_sorted[j].label:
                    if iou(det.bbox, detections_sorted[j].bbox) > iou_threshold:
                        used[j] = True

        return filtered
