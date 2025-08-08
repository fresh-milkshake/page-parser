import sys
import json
import click
from pathlib import Path

from src.common.logging import setup_logging, get_logger
from src.pipeline import pipeline
from src.config.paths import DEFAULT_LOG_FILE, DEFAULT_SETTINGS_FILE

logger = get_logger(__name__)


@click.command()
@click.argument("document_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.argument("output_json", type=click.Path(dir_okay=False, writable=True))
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
@click.option(
    "--settings-file",
    default=DEFAULT_SETTINGS_FILE,
    type=click.Path(dir_okay=False),
    help="Path to settings file.",
)
def main(
    document_path: str,
    model_path: str,
    output_dir: str,
    output_json: str,
    log_level: str,
    log_file: str,
    settings_file: str,
) -> None:
    """
    CLI for running the document analysis pipeline and saving results to a JSON file.
    """

    log_file_path = Path(log_file) if log_file else None
    setup_logging(log_level=log_level, log_file=log_file_path)

    logger.info("Starting page-parser CLI")
    logger.info(f"Document path: {document_path}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output JSON: {output_json}")
    logger.info(f"Settings file: {settings_file}")

    try:
        results = pipeline(
            document_path=document_path,
            model_path=model_path,
            output_dir=output_dir,
            settings_file=Path(settings_file),
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
