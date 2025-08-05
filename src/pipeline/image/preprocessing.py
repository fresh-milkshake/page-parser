from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

from src.common.aliases import Color, Rectangle
from src.common.logging import get_logger

logger = get_logger(__name__)


def fill_regions_with_color(
    image_path: Path,
    regions: list[Rectangle],
    output_path: Path,
    color: Color = (255, 255, 255),
) -> None:
    """
    Fill specified rectangular regions in an image with a solid color.

    Args:
        image_path: Path to the input image.
        regions: List of rectangles (x1, y1, x2, y2) to fill.
        output_path: Path to save the modified image.
        color: RGB color tuple to fill the regions with.
    """
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            for rect in regions:
                draw.rectangle(rect, fill=color)  # type: ignore
            img.save(output_path)
        logger.info(
            f"Filled {len(regions)} regions in '{image_path.stem}' and saved to '{output_path.stem}'."
        )
    except Exception as exc:
        logger.error(f"Failed to fill regions in image '{image_path}': {exc}")
        raise


# THIS FUNCTION IS FOR USAGE IN SHOWCASE
def fill_regions_with_color_to_numpy(
    image_path: Path,
    regions: list[Rectangle],
    color: Color = (255, 255, 255),
) -> np.ndarray:
    """
    Fill specified rectangular regions in an image with a solid color.

    Args:
        image_path: Path to the input image.
        regions: List of rectangles (x1, y1, x2, y2) to fill.
        output_path: Path to save the modified image.
        color: RGB color tuple to fill the regions with.
    """
    try:
        with Image.open(image_path) as img:
            for rect in regions:
                draw = ImageDraw.Draw(img)
                draw.rectangle(rect, fill=color)  # type: ignore
        logger.info(f"Filled {len(regions)} regions in '{image_path.stem}'.")
        return np.array(img)
    except Exception as exc:
        logger.error(f"Failed to fill regions in image '{image_path}': {exc}")
        raise


def extract_regions(
    image_path: Path,
    regions: list[Rectangle],
    output_dir: Path,
) -> tuple[Path, ...] | None:
    """
    Extract specified rectangular regions from an image and save each as a separate file.

    Args:
        image_path: Path to the input image.
        regions: List of rectangles (x1, y1, x2, y2) to extract.
        output_dir: Path to the output directory.

    Returns:
        Tuple of Paths to the saved region images.
    """
    output_paths: list[Path] = []
    try:
        with Image.open(image_path) as img:
            for idx, rect in enumerate(regions):
                region = img.crop(rect)
                output_path = (
                    output_dir / f"{image_path.stem}_region_{idx}{image_path.suffix}"
                )
                region.save(output_path)
                output_paths.append(output_path)
        logger.info(
            f"Extracted {len(regions)} regions from '{image_path.stem}' to '{output_dir.stem}/{image_path.stem}'."
        )
        return tuple(output_paths)
    except Exception as exc:
        logger.error(f"Failed to extract regions from image '{image_path}': {exc}")
        return None


# THIS FUNCTION IS FOR USAGE IN SHOWCASE
def extract_regions_to_numpy(
    image_path: Path,
    regions: list[Rectangle],
) -> list[np.ndarray]:
    """
    Extract specified rectangular regions from an image and return them as numpy arrays.
    """
    with Image.open(image_path) as img:
        return [np.array(img.crop(rect)) for rect in regions]