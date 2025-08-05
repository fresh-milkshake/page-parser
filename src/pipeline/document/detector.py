from pathlib import Path
from typing import List
from dataclasses import dataclass
import cv2
from ultralytics import YOLO

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    """
    Data class representing a single detection result.

    Attributes:
        label (int): The class index of the detected object.
        label_name (str): The class name of the detected object.
        bbox (List[float]): Bounding box coordinates in pixel values [x1, y1, x2, y2].
        confidence (float): Confidence score of the detection.
    """

    label: int
    label_name: str
    bbox: List[float]
    confidence: float


class Detector:
    """
    Detector for running YOLO detection and returning detection results.

    Attributes:
        model_path (str): Path to the YOLO model file.
        line_width (int): Width of the bounding box lines.
        font_size (float): Font size for labels.
    """

    def __init__(
        self,
        model_path: str,
        line_width: int = 2,
        font_size: float = 0.5,
    ) -> None:
        """
        Initialize the Detector with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file.
            line_width (int, optional): Width of the bounding box lines. Defaults to 2.
            font_size (float, optional): Font size for labels. Defaults to 0.5.
        """
        logger.info(f"Initializing YOLO detector with model: {model_path}")
        self.model_path = model_path
        self.line_width = line_width
        self.font_size = font_size
        self.model = YOLO(self.model_path)
        logger.info("YOLO detector initialized successfully")

    def parse_layout(
        self,
        image_path: Path,
    ) -> List[DetectionResult]:
        """
        Run detection on an image and return detection results.

        Args:
            image_path (str): Path to the input image.

        Returns:
            List[DetectionResult]: List of detection results.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        logger.debug(f"Running layout detection on: {image_path}")

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"Could not load image at {image_path}")
            raise FileNotFoundError(f"Could not load image at {image_path}")

        logger.debug(f"Image loaded successfully, shape: {img.shape}")
        result = self.model.predict(img)[0]
        height, width = img.shape[:2]

        detections: List[DetectionResult] = []
        for label, box, conf in zip(
            result.boxes.cls.tolist(),  # type: ignore
            result.boxes.xyxyn.tolist(),  # type: ignore
            result.boxes.conf.tolist(),  # type: ignore
        ):  # type: ignore[attr-defined]
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
        return detections
