from pathlib import Path
from src.pipeline.image import ChartSummarizer
from src.pipeline.utils import load_rgb_image
from src.common.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    chart_summarizer = ChartSummarizer(
        model_name="gemma3",
        api_key="dummy",
        base_url="http://localhost:11434/v1",
        retries=3,
        timeout=10,
    )

    image = load_rgb_image(Path("data/page_30.png"))

    result = chart_summarizer.summarize_charts_from_page(
        image=image,
        prompt="Summarize the chart in the image.",
        extra_context="The chart is a bar chart.",
    )

    print(result)
    
    
if __name__ == "__main__":
    main()
