from typing import List

from doclayout_yolo import YOLO
import numpy as np

from src.common.logging import get_logger
from src.config.settings import Settings

from .base import BaseDetectionBackend, DetectionResult


logger = get_logger(__name__)


class YoloDoclaynetBackend(BaseDetectionBackend):
    """
    Detector for running YOLO detection and returning detection results.

    Attributes:
        model_path (str): Path to the YOLO model file.
    """

    def __init__(
        self,
        model_path: str,
        _: Settings,
    ) -> None:
        """
        Initialize the Detector with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        logger.info(f"Initializing YoloDoclaynetBackend with model: {model_path}")
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        logger.info("YoloDoclaynetBackend initialized successfully")

    def detect(
        self,
        image: np.ndarray,
    ) -> List[DetectionResult]:
        """
        Run detection on an image and return detection results.

        Args:
            image (np.ndarray): Image to detect objects in.

        Returns:
            List[DetectionResult]: List of detection results.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        logger.debug(f"Running layout detection on shape: {image.shape}")

        result = self.model.predict(image)[0]
        height, width = image.shape[:2]

        detections: List[DetectionResult] = []
        for label, box, conf in zip(
            result.boxes.cls.tolist(),
            result.boxes.xyxyn.tolist(),
            result.boxes.conf.tolist(),
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
