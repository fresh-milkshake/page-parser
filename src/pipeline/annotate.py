from typing import List
import cv2
from ultralytics.utils.plotting import Annotator, Colors

from src.common.logging import get_logger
from src.pipeline.detector import DetectionResult

logger = get_logger(__name__)


def annotate_image(
    image_path: str,
    detections: List[DetectionResult],
    output_path: str,
    line_width: int = 2,
    font: str = "Monospace.ttf",
    font_size: int = 10,
) -> None:
    """
    Annotate an image with bounding boxes and labels.

    Args:
        image_path (str): Path to the input image.
        detections (List[DetectionResult]): List of detection results to annotate.
        output_path (str): Path to save the annotated image.
        line_width (int, optional): Width of the bounding box lines. Defaults to 2.
        font_family (str, optional): Font family for labels. Defaults to "Monospace".
        font_size (float, optional): Font size for labels. Defaults to 0.5.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    logger.info(f"Annotating image: {image_path}")
    logger.info(f"Output path: {output_path}")
    logger.debug(f"Annotating {len(detections)} detections")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        raise FileNotFoundError(f"Could not load image at {image_path}")

    annotator = Annotator(img, line_width=line_width, font_size=font_size, font=font)
    colors = Colors()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox
        logger.debug(f"Annotating detection {idx + 1}: {det.label_name} at {det.bbox}")
        annotator.box_label(
            [x1, y1, x2, y2],
            det.label_name,
            color=colors(det.label, bgr=True),
        )

    annotator.save(output_path)
    logger.info(f"Successfully saved annotated image to {output_path}")
