import cv2
import numpy as np
import pytesseract

from src.common.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_image(
    image_path: str,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    extra_config: str | None = None,
) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_path (str): Path to the input image.
        lang (str): Language(s) for OCR. Defaults to "eng".
        psm (int): Page segmentation mode for Tesseract. Defaults to 6.
        oem (int): OCR Engine mode. Defaults to 3.
        extra_config (Optional[str]): Additional Tesseract config options.

    Returns:
        str: Extracted text from the image.

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    logger.debug(f"Extracting text from image: {image_path}")
    logger.debug(f"OCR settings: lang={lang}, psm={psm}, oem={oem}")

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        raise FileNotFoundError(f"Could not load image at {image_path}")

    config = f"--oem {oem} --psm {psm}"
    if extra_config:
        config = f"{config} {extra_config}"

    logger.debug(f"Tesseract config: {config}")
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    result = text.strip()

    logger.debug(f"Extracted {len(result)} characters of text")
    return result


# THIS FUNCTION IS FOR USAGE IN SHOWCASE
def extract_text_from_numpy(
    image: np.ndarray,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    extra_config: str | None = None,
) -> str:
    """
    Extract text from a numpy array using Tesseract OCR.

    Args:
        image (np.ndarray): Numpy array of the image.
        lang (str): Language(s) for OCR. Defaults to "eng".
        psm (int): Page segmentation mode for Tesseract. Defaults to 6.
        oem (int): OCR Engine mode. Defaults to 3.
        extra_config (Optional[str]): Additional Tesseract config options.

    Returns:
        str: Extracted text from the image.
    """
    logger.debug(f"OCR settings: lang={lang}, psm={psm}, oem={oem}")

    config = f"--oem {oem} --psm {psm}"
    if extra_config:
        config = f"{config} {extra_config}"

    logger.debug(f"Tesseract config: {config}")
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    result = text.strip()

    logger.debug(f"Extracted {len(result)} characters of text")
    return result