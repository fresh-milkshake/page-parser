import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile

import cv2
import numpy as np

from src.common.logging import get_logger
from src.pipeline.convert import pdf_to_png
from src.pipeline.detector import Detector, DetectionResult
from src.pipeline.image_summarizer import ChartSummarizer
from src.pipeline.text_extraction import extract_text_from_image

logger = get_logger(__name__)


def cut_charts_from_image(
    image_path: Path,
    detections: List[DetectionResult],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    Crop chart and diagram regions from an image using bounding boxes.

    Args:
        image_path (str): Path to the input image.
        detections (List[DetectionResult]): List of detection results with bounding boxes.
        output_dir (str): Directory to save cropped chart images.

    Returns:
        List[Dict[str, Any]]: List of dicts with file path and bbox for each cropped chart.
    """
    logger.info(f"Cutting charts from image: {image_path}")
    logger.debug(f"Found {len(detections)} chart detections")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        raise FileNotFoundError(f"Could not load image at {image_path}")

    cropped_info: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det.bbox)
        crop = img[y1:y2, x1:x2]
        crop_filename = (
            Path(output_dir) / f"{Path(image_path).stem}_chart_{idx + 1}.png"
        )
        cv2.imwrite(str(crop_filename), crop)
        logger.debug(f"Saved chart {idx + 1} to {crop_filename}")
        cropped_info.append(
            {
                "path": str(crop_filename),
                "bbox": (x1, y1, x2, y2),
                "label": det.label_name,
            }
        )

    logger.info(f"Successfully cut {len(cropped_info)} charts from image")
    return cropped_info


def get_non_chart_regions(
    image_shape: tuple,
    chart_bboxes: List[tuple],
    min_area: int = 1000,
) -> List[tuple]:
    """
    Get bounding boxes for non-chart (background) regions in the image.

    Args:
        image_shape (tuple): Shape of the image (height, width, channels).
        chart_bboxes (List[tuple]): List of chart bounding boxes.
        min_area (int): Minimum area for a region to be considered.

    Returns:
        List[tuple]: List of bounding boxes for non-chart regions.
    """
    height, width = image_shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    for x1, y1, x2, y2 in chart_bboxes:
        mask[y1:y2, x1:x2] = 0

    # Find connected components in the mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    non_chart_bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            non_chart_bboxes.append((x, y, x + w, y + h))
    return non_chart_bboxes


def cut_text_regions_from_image(
    image_path: Path,
    non_chart_bboxes: List[tuple],
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    Crop non-chart (text) regions from an image.

    Args:
        image_path (str): Path to the input image.
        non_chart_bboxes (List[tuple]): List of bounding boxes for text regions.
        output_dir (str): Directory to save cropped text images.

    Returns:
        List[Dict[str, Any]]: List of dicts with file path and bbox for each cropped text region.
    """
    logger.info(f"Cutting text regions from image: {image_path}")
    logger.debug(f"Found {len(non_chart_bboxes)} text regions")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        raise FileNotFoundError(f"Could not load image at {image_path}")

    cropped_info: List[Dict[str, Any]] = []
    for idx, bbox in enumerate(non_chart_bboxes):
        x1, y1, x2, y2 = bbox
        crop = img[y1:y2, x1:x2]
        crop_filename = Path(output_dir) / f"{Path(image_path).stem}_text_{idx + 1}.png"
        cv2.imwrite(str(crop_filename), crop)
        logger.debug(f"Saved text region {idx + 1} to {crop_filename}")
        cropped_info.append(
            {
                "path": str(crop_filename),
                "bbox": bbox,
            }
        )

    logger.info(f"Successfully cut {len(cropped_info)} text regions from image")
    return cropped_info


def filter_chart_detections(detections: List[DetectionResult]) -> List[DetectionResult]:
    """
    Filter detections to only include chart/picture-related elements.

    Args:
        detections (List[DetectionResult]): List of all detections.

    Returns:
        List[DetectionResult]: Filtered list containing only chart/picture detections.
    """
    chart_labels = {"picture", "figure", "chart", "diagram"}
    return [det for det in detections if det.label_name.lower() in chart_labels]


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
    chart_crops = cut_charts_from_image(
        image_path=image_path,
        detections=chart_detections,
        output_dir=str(cropped_dir),
    )

    chart_elements: List[Dict[str, Any]] = []
    for idx, chart in enumerate(chart_crops):
        logger.debug(f"Summarizing chart {idx + 1}/{len(chart_crops)}")
        summary = summarizer.summarize_charts_from_page(
            image_path=chart["path"],
            prompt=prompt,
            extra_context=extra_context,
        )
        chart_elements.append(
            {
                "type": "chart",
                "label": chart["label"].lower(),
                "summary": summary,
                "bbox": chart["bbox"],
            }
        )

    logger.info(f"Successfully processed {len(chart_elements)} chart elements")
    return chart_elements


def process_text_elements(
    image_path: Path,
    chart_bboxes: List[tuple],
    temp_path: Path,
    page_number: int,
    ocr_lang: str,
) -> List[Dict[str, Any]]:
    """
    Process text elements: fill chart regions with white, then extract text from the entire image.

    Args:
        image_path (Path): Path to the input image.
        chart_bboxes (List[tuple]): Chart bounding boxes to fill with white.
        temp_path (Path): Temporary directory path.
        page_number (int): Current page number.
        ocr_lang (str): Language for Tesseract OCR.

    Returns:
        List[Dict[str, Any]]: List of text elements with extracted content.
    """
    logger.info(f"Processing text elements for page {page_number}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Could not load image at {image_path}")
        raise FileNotFoundError(f"Could not load image at {image_path}")

    # Create a copy of the image to modify
    img_processed = img.copy()

    # Fill all chart bounding boxes with white
    for x1, y1, x2, y2 in chart_bboxes:
        img_processed[y1:y2, x1:x2] = [255, 255, 255]  # White color in BGR
        logger.debug(f"Filled chart bbox ({x1}, {y1}, {x2}, {y2}) with white")

    # Save the processed image with chart regions filled
    processed_image_path = temp_path / f"text_processed_{page_number}.png"
    cv2.imwrite(str(processed_image_path), img_processed)
    logger.debug(f"Saved processed image to {processed_image_path}")

    # Extract text from the entire processed image
    logger.debug("Extracting text from processed image")
    text = extract_text_from_image(
        image_path=str(processed_image_path),
        lang=ocr_lang,
    )

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


def process_single_page(
    img_path: Path,
    detector: Detector,
    summarizer: ChartSummarizer,
    temp_path: Path,
    prompt: Optional[str] = None,
    extra_context: Optional[str] = None,
    ocr_lang: str = "eng",
) -> Dict[str, Any]:
    """
    Process a single page: detect elements, extract charts and text.

    Args:
        img_path (Path): Path to the page image.
        detector (Detector): YOLO detector instance.
        summarizer (ChartSummarizer): Chart summarizer instance.
        temp_path (Path): Temporary directory path.
        prompt (Optional[str]): Custom prompt for summarization.
        extra_context (Optional[str]): Extra context for summarization.
        ocr_lang (str): Language for Tesseract OCR.

    Returns:
        Dict[str, Any]: Page data with elements.
    """
    page_number = int(img_path.stem.split("_")[1])
    logger.info(f"Processing page {page_number} from {img_path}")

    # Detect and filter chart elements
    logger.debug("Running layout detection")
    detections = detector.parse_layout(img_path)
    logger.info(f"Found {len(detections)} total detections")

    chart_detections = filter_chart_detections(detections)
    logger.info(f"Filtered to {len(chart_detections)} chart detections")
    chart_bboxes = [tuple(map(int, det.bbox)) for det in chart_detections]

    # Process chart elements
    chart_elements = process_chart_elements(
        image_path=img_path,
        chart_detections=chart_detections,
        temp_path=temp_path,
        page_number=page_number,
        summarizer=summarizer,
        prompt=prompt,
        extra_context=extra_context,
    )

    # Process text elements
    text_elements = process_text_elements(
        image_path=img_path,
        chart_bboxes=chart_bboxes,
        temp_path=temp_path,
        page_number=page_number,
        ocr_lang=ocr_lang,
    )

    # Combine all elements
    page_elements = chart_elements + text_elements
    logger.info(
        f"Page {page_number} completed with {len(page_elements)} total elements"
    )

    return {
        "page_number": page_number,
        "elements": page_elements,
    }


def pipeline(
    document_path: str,
    model_path: str,
    output_dir: str,
    ollama_model: str = "gemma3",
    ollama_host: str = "http://localhost:11434",
    zoom_x: float = 2.0,
    zoom_y: float = 2.0,
    colorspace: str = "rgb",
    font_size: float = 0.5,
    line_width: int = 2,
    prompt: Optional[str] = None,
    extra_context: Optional[str] = None,
    ocr_lang: str = "eng",
) -> List[Dict[str, Any]]:
    """
    Full pipeline: convert PDF to PNG, detect charts, extract text, and summarize.

    Args:
        document_path (str): Path to the input PDF document.
        model_path (str): Path to the YOLO model file.
        output_dir (str): Directory to save outputs.
        ollama_model (str): Name of the Ollama vision model.
        ollama_host (str): Host address for Ollama server.
        zoom_x (float): Horizontal zoom for PDF rendering.
        zoom_y (float): Vertical zoom for PDF rendering.
        colorspace (str): Colorspace for PNG output.
        font_size (float): Font size for annotation.
        line_width (int): Line width for annotation.
        prompt (Optional[str]): Custom prompt for summarization.
        extra_context (Optional[str]): Extra context for summarization.
        ocr_lang (str): Language for Tesseract OCR.

    Returns:
        List[Dict[str, Any]]: List of dicts for each page, each containing a list of elements (charts and text).
    """
    logger.info("Starting document processing pipeline")
    logger.info(f"Input document: {document_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Ollama model: {ollama_model}")
    logger.info(f"OCR language: {ocr_lang}")

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
            zoom_x=zoom_x,
            zoom_y=zoom_y,
            colorspace=colorspace,
        )
        logger.info(f"Converted PDF to {len(image_paths)} PNG images")

        # Initialize models
        logger.info("Initializing detection and summarization models")
        detector = Detector(
            model_path=model_path, line_width=line_width, font_size=font_size
        )
        summarizer = ChartSummarizer(model_name=ollama_model, ollama_host=ollama_host)
        logger.info("Models initialized successfully")

        # Process each page
        results: List[Dict[str, Any]] = []
        for idx, img_path in enumerate(image_paths):
            logger.info(f"Processing page {idx + 1}/{len(image_paths)}")
            page_result = process_single_page(
                img_path=img_path,
                detector=detector,
                summarizer=summarizer,
                temp_path=temp_path,
                prompt=prompt,
                extra_context=extra_context,
                ocr_lang=ocr_lang,
            )
            results.append(page_result)

            with open(Path(output_dir) / f"page_{idx + 1}.json", "w") as f:
                json.dump(page_result, f)

    logger.info("Pipeline completed successfully")
    logger.info(
        f"Processed {len(results)} pages with total elements: {sum(len(page['elements']) for page in results)}"
    )
    return results
