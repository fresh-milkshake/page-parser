import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, Colors

from src.common.logging import get_logger
from src.pipeline.document.detector import DetectionResult

logger = get_logger(__name__)


class Fonts:
    Monospace = "Monospace.ttf"
    Arial = "Arial.ttf"
    TimesNewRoman = "TimesNewRoman.ttf"
    CourierNew = "CourierNew.ttf"
    Calibri = "Calibri.ttf"
    Cambria = "Cambria.ttf"
    Consolas = "Consolas.ttf"
    Georgia = "Georgia.ttf"
    SegoeUI = "SegoeUI.ttf"
    Tahoma = "Tahoma.ttf"
    Verdana = "Verdana.ttf"
    Webdings = "Webdings.ttf"
    Wingdings = "Wingdings.ttf"
    ZapfDingbats = "ZapfDingbats.ttf"


def annotate_image(
    image_path: str,
    detections: list[DetectionResult],
    line_width: int = 2,
    font: str = Fonts.Monospace,
    font_size: int = 10,
    autosave: bool = False,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Annotate an image with bounding boxes and labels and return the annotated image as a numpy array.

    Args:
        image_path (str): Path to the input image.
        detections (list[DetectionResult]): List of detection results to annotate.
        line_width (int, optional): Width of the bounding box lines. Defaults to 2.
        font (str, optional): Font for labels. Defaults to "Monospace.ttf".
        font_size (int, optional): Font size for labels. Defaults to 10.
        autosave (bool, optional): Whether to save the annotated image. Defaults to False.
        save_path (str, optional): Path to save the annotated image. Defaults to None.

    Returns:
        np.ndarray: Annotated image as a numpy array.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    logger.info(f"Annotating image: {image_path}")
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

    annotated_img = annotator.result()
    logger.info("Successfully annotated image")

    if autosave:
        if save_path is None:
            save_path = image_path.replace(".png", "_annotated.png")
        cv2.imwrite(save_path, annotated_img)
        logger.info(f"Successfully saved annotated image to {save_path}")

    return annotated_img
