from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, TypeAlias

import numpy as np

from src.config.settings import Settings


@dataclass
class DetectionResult:
    """
    Data class representing a single detection result.
    Kept for backward compatibility.

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


DetectionResults: TypeAlias = List[DetectionResult]


class BaseDetectionBackend(ABC):
    """
    Base class for detection backends.
    """

    @abstractmethod
    def __init__(self, model_path: str, settings: Settings) -> None:
        """
        Initialize the detection backend.
        """
        pass

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
    ) -> DetectionResults:
        """
        Detect objects in an image and return detection results.

        Args:
            image (np.ndarray): Image to detect objects in.

        Returns:
            List[DetectionResult]: List of detection results.
        """
        pass
