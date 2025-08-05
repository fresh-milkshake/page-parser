from typing import Any, Dict, List, Optional, cast
import tempfile
from pathlib import Path
import json
import cv2

from src.config.settings import get_settings
from src.common.logging import get_logger
from src.pipeline.document import (
    pdf_to_png,
    Detector,
    DetectionResult,
    extract_text_from_image,
)
from src.pipeline.image import ChartSummarizer
from src.pipeline.image import extract_regions, fill_regions_with_color
from src.pipeline.utils import filter_detections
from src.common.aliases import Rectangle

logger = get_logger(__name__)


def pipeline(
    document_path: str,
    model_path: str,
    output_dir: str,
    settings_file: Path,
    page_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Full pipeline: convert PDF to PNG, detect charts, extract text, and summarize.

    Args:
        document_path (str): Path to the input PDF document.
        model_path (str): Path to the YOLO model file.
        output_dir (str): Directory to save outputs.
        settings_file (Path): Path to the settings file.

    Returns:
        List[Dict[str, Any]]: List of dicts for each page, each containing a list of elements (charts and text).
    """
    logger.info("Starting document processing pipeline")

    settings = get_settings(settings_file)

    if not settings.vision.provider.is_configured():
        raise ValueError(
            "Vision provider is not configured: "
            f"{settings.vision.provider.unsafe_repr()}",
        )

    model_name = settings.vision.provider.model
    base_url = settings.vision.provider.base_url
    api_key = settings.vision.provider.api_key

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create temporary directory for intermediate data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.debug(f"Created temporary directory: {temp_path}")

        # Convert PDF to PNG in temporary directory
        logger.info("Converting PDF to PNG images")
        image_paths = pdf_to_png(
            pdf_path=document_path,
            output_dir=str(temp_path),
            zoom_x=settings.processing.zoom_factor,
            zoom_y=settings.processing.zoom_factor,
        )
        logger.info(f"Converted PDF to {len(image_paths)} PNG images")

        # Initialize models
        logger.info("Initializing detection and summarization models")
        detector = Detector(model_path=model_path)
        summarizer = ChartSummarizer(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key or "",
            retries=settings.vision.retries,
            timeout=settings.vision.timeout,
        )
        logger.info("Models initialized successfully")

        # Process each page
        results: List[Dict[str, Any]] = []
        for idx, img_path in enumerate(image_paths):
            if page_limit is not None and idx >= page_limit:
                logger.info(f"Page limit of {page_limit} reached, stopping processing.")
                break

            logger.info(f"Processing page {idx + 1}/{len(image_paths)}")
            page_result = process_single_page(
                img_path=img_path,
                detector=detector,
                summarizer=summarizer,
                temp_path=temp_path,
                ocr_lang=settings.processing.ocr_lang,
                charts_labels=settings.filtration.chart_labels,
            )
            results.append(page_result)

            with open(Path(output_dir) / f"page_{idx + 1}.json", "w") as f:
                json.dump(page_result, f)

    logger.info("Pipeline completed successfully")
    logger.info(
        f"Processed {len(results)} pages with total elements: {sum(len(page['elements']) for page in results)}"
    )
    return results


def process_single_page(
    img_path: Path,
    detector: Detector,
    summarizer: ChartSummarizer,
    temp_path: Path,
    charts_labels: List[str],
    ocr_lang: str,
) -> Dict[str, Any]:
    """
    Process a single page image: detect elements, extract chart and text data.

    Args:
        img_path (Path): Path to the page image.
        detector (Detector): Detector instance for layout detection.
        summarizer (ChartSummarizer): Summarizer for chart elements.
        temp_path (Path): Temporary directory for intermediate files.
        ocr_lang (str): Language for OCR.

    Returns:
        Dict[str, Any]: Dictionary containing page number and extracted elements.
    """

    page_number = int(img_path.stem.split("_")[1]) if "_" in img_path.stem else 1

    detections = detector.parse_layout(img_path)
    chart_detections = filter_detections(detections, charts_labels)

    chart_elements = process_chart_elements(
        image_path=img_path,
        chart_detections=chart_detections,
        temp_path=temp_path,
        page_number=page_number,
        summarizer=summarizer,
    )

    text_elements = process_text_elements(
        image_path=img_path,
        chart_bboxes=chart_detections,
        temp_path=temp_path,
        ocr_lang=ocr_lang,
        page_number=page_number,
    )

    elements = chart_elements + text_elements

    return {
        "page_number": page_number,
        "elements": elements,
    }


def process_chart_elements(
    image_path: Path,
    chart_detections: List[DetectionResult],
    temp_path: Path,
    page_number: int,
    summarizer: ChartSummarizer,
    prompt: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process chart elements: crop, summarize, and return structured data.

    Args:
        image_path (Path): Path to the input image.
        chart_detections (List[DetectionResult]): Filtered chart detections.
        temp_path (Path): Temporary directory path.
        page_number (int): Current page number.
        summarizer (ChartSummarizer): Chart summarizer instance.
        prompt (Optional[str]): Custom prompt for summarization.
        extra_context (Optional[str]): Extra context for summarization.

    Returns:
        List[Dict[str, Any]]: List of chart elements with summaries.
    """
    logger.info(f"Processing chart elements for page {page_number}")

    if not chart_detections:
        logger.info("No chart detections found, skipping chart processing")
        return []

    cropped_dir = temp_path / f"cropped_{page_number}"
    cropped_dir.mkdir(exist_ok=True)

    # Extract regions and create the expected format
    region_paths = extract_regions(
        image_path=image_path,
        regions=[
            cast(Rectangle, tuple(detection.bbox)) for detection in chart_detections
        ],
        output_dir=cropped_dir,
    )

    if region_paths is None:
        logger.warning("Failed to extract chart regions")
        return []

    chart_elements: List[Dict[str, Any]] = []
    for idx, (detection, region_path) in enumerate(zip(chart_detections, region_paths)):
        logger.debug(f"Summarizing chart {idx + 1}/{len(region_paths)}")
        summary = summarizer.summarize_charts_from_page(
            image_path=str(region_path),
            prompt=prompt,
            extra_context=extra_context,
        )
        chart_elements.append(
            {
                "type": "chart",
                "label": detection.label_name.lower(),
                "summary": summary,
                "bbox": detection.bbox,
            }
        )

    logger.info(f"Successfully processed {len(chart_elements)} chart elements")
    return chart_elements


def process_text_elements(
    image_path: Path,
    chart_bboxes: List[DetectionResult],
    temp_path: Path,
    page_number: int,
    ocr_lang: str,
) -> List[Dict[str, Any]]:
    """
    Process text elements: fill chart regions with white, then extract text from the entire image.

    Args:
        image_path (Path): Path to the input image.
        chart_bboxes (List[DetectionResult]): Chart bounding boxes to fill with white.
        temp_path (Path): Temporary directory path.
        page_number (int): Current page number.
        ocr_lang (str): Language for Tesseract OCR.

    Returns:
        List[Dict[str, Any]]: List of text elements with extracted content.
    """
    logger.info(f"Processing text elements for page {page_number}")

    # Save the processed image with chart regions filled
    processed_image_path = temp_path / f"text_processed_{page_number}.png"

    # Fill chart regions with white using the dedicated function
    fill_regions_with_color(
        image_path=image_path,
        regions=[cast(Rectangle, tuple(detection.bbox)) for detection in chart_bboxes],
        output_path=processed_image_path,
        color=(255, 255, 255),  # White color
    )

    logger.debug(f"Saved processed image to {processed_image_path}")

    # Extract text from the entire processed image
    logger.debug("Extracting text from processed image")
    text = extract_text_from_image(
        image_path=str(processed_image_path),
        lang=ocr_lang,
    )

    # Get image dimensions for text bbox
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        raise FileNotFoundError(f"Could not load image at {image_path}")

    text_elements: List[Dict[str, Any]] = []
    if text.strip():
        # Use entire image dimensions as bbox for text
        height, width = img.shape[:2]
        text_elements.append(
            {
                "type": "text",
                "text": text,
                "bbox": (0, 0, width, height),
            }
        )
        logger.info("Successfully extracted text from processed image")
    else:
        logger.info("No text extracted from processed image")

    logger.info(f"Successfully processed {len(text_elements)} text elements")
    return text_elements
