import sys
import json
from typing import Optional
import click

from src.common.logging import setup_logging, get_logger
from src.pipeline import pipeline
from src.config.paths import DEFAULT_LOG_FILE

logger = get_logger(__name__)


@click.command()
@click.argument("document_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.argument("output_json", type=click.Path(dir_okay=False, writable=True))
@click.option(
    "--ollama-model",
    default="gemma3",
    show_default=True,
    help="Ollama vision model name.",
)
@click.option(
    "--ollama-host",
    default="http://localhost:11434",
    show_default=True,
    help="Ollama server host.",
)
@click.option(
    "--zoom-x",
    default=2.0,
    show_default=True,
    type=float,
    help="Horizontal zoom for PDF rendering.",
)
@click.option(
    "--zoom-y",
    default=2.0,
    show_default=True,
    type=float,
    help="Vertical zoom for PDF rendering.",
)
@click.option(
    "--colorspace",
    default="rgb",
    show_default=True,
    type=click.Choice(["rgb", "gray"]),
    help="Colorspace for PNG output.",
)
@click.option(
    "--font-size",
    default=0.5,
    show_default=True,
    type=float,
    help="Font size for annotation.",
)
@click.option(
    "--line-width",
    default=2,
    show_default=True,
    type=int,
    help="Line width for annotation.",
)
@click.option(
    "--prompt", default=None, type=str, help="Custom prompt for summarization."
)
@click.option(
    "--extra-context", default=None, type=str, help="Extra context for summarization."
)
@click.option(
    "--ocr-lang",
    default="eng",
    show_default=True,
    type=str,
    help="Language for Tesseract OCR.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level.",
)
@click.option(
    "--log-file",
    default=DEFAULT_LOG_FILE,
    type=click.Path(dir_okay=False),
    help="Path to log file (optional).",
)
def main(
    document_path: str,
    model_path: str,
    output_dir: str,
    output_json: str,
    ollama_model: str,
    ollama_host: str,
    zoom_x: float,
    zoom_y: float,
    colorspace: str,
    font_size: float,
    line_width: int,
    prompt: Optional[str],
    extra_context: Optional[str],
    ocr_lang: str,
    log_level: str,
    log_file: Optional[str],
) -> None:
    """
    CLI for running the document analysis pipeline and saving results to a JSON file.

    Args:
        document_path (str): Path to the input PDF document.
        model_path (str): Path to the YOLO model file.
        output_dir (str): Directory to save outputs.
        output_json (str): Path to the output JSON file.
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
        log_level (str): Logging level.
        log_file (Optional[str]): Path to log file.
    """
    # Setup logging
    from pathlib import Path

    log_file_path = Path(log_file) if log_file else None
    setup_logging(log_level=log_level, log_file=log_file_path)

    logger.info("Starting page-parser CLI")
    logger.info(f"Document path: {document_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output JSON: {output_json}")

    try:
        results = pipeline(
            document_path=document_path,
            model_path=model_path,
            output_dir=output_dir,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            zoom_x=zoom_x,
            zoom_y=zoom_y,
            colorspace=colorspace,
            font_size=font_size,
            line_width=line_width,
            prompt=prompt,
            extra_context=extra_context,
            ocr_lang=ocr_lang,
        )

        logger.info(f"Saving results to {output_json}")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("Pipeline completed successfully")
        click.echo(f"Pipeline completed successfully. Output written to {output_json}.")
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
