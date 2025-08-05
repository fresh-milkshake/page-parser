from typing import List

from src.pipeline.document.detector import DetectionResult


def filter_detections(
    detections: List[DetectionResult], labels: List[str]
) -> List[DetectionResult]:
    """
    Filter detections to only include elements with specified labels.

    Args:
        detections (List[DetectionResult]): List of all detections.
        labels (List[str]): List of lowercased labels to filter by.

    Returns:
        List[DetectionResult]: Filtered list containing only detections with specified labels.
    """
    return [det for det in detections if det.label_name.lower() in labels]
