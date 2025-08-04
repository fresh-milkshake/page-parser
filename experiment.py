import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors

from src.common.logging import setup_logging, get_logger

logger = get_logger(__name__)

# Make sure to install ultralytics:
# pip install ultralytics
# If you have issues, see: https://docs.ultralytics.com/quickstart/#install-ultralytics

# Download the pretrained model from HuggingFace if not already present.
# Example: yolov8n-doclaynet.pt, yolov8s-doclaynet.pt, or yolov8m-doclaynet.pt
# You can download from: https://huggingface.co/keremberke/yolov8n-doclaynet or similar

# Setup logging
setup_logging()

# Set your model and image path
your_model_path = "yolov12l-doclaynet.pt"  # Change to yolov8s-doclaynet.pt or yolov8m-doclaynet.pt if desired
your_image_path = "image.png"

logger.info("Starting experiment")
logger.info(f"Model path: {your_model_path}")
logger.info(f"Image path: {your_image_path}")

# Read the image
img = cv2.imread(your_image_path, cv2.IMREAD_COLOR)
if img is None:
    logger.error(f"Could not load image at {your_image_path}")
    raise FileNotFoundError(f"Could not load image at {your_image_path}")

logger.info(f"Image loaded successfully, shape: {img.shape}")

# Load the model
logger.info("Loading YOLO model")
model = YOLO(your_model_path)
logger.info("Model loaded successfully")

# Run detection
logger.info("Running detection")
result = model.predict(img)[0]
logger.info(f"Detection completed: {result}")

# Visualize and save the results
logger.info("Creating annotations")

height, width = img.shape[:2]
line_width = 2
font_size = 10  # Fixed: changed from float to int
colors = Colors()
annotator = Annotator(img, line_width=line_width, font_size=font_size)

if result.boxes is not None:  # Fixed: added null check
    for label, box in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist()):
        label = int(label)
        logger.debug(f"Annotating detection: {result.names[label]} at {box}")
        annotator.box_label(
            [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
            result.names[label],
            color=colors(label, bgr=True),
        )

annotated_path = os.path.join(
    os.path.dirname(your_image_path), "annotated-" + os.path.basename(your_image_path)
)
annotator.save(annotated_path)
logger.info(f"Annotated image saved to {annotated_path}")
print(f"Annotated image saved to {annotated_path}")
