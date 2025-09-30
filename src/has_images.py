import fitz  # PyMuPDF
from src.utils import get_logger
# Get logger through the utility function
logger = get_logger(__name__)

def pdf_contains_images(path: str) -> bool:
    image_count = 0
    try:
        with fitz.open(path) as doc:
            for page in doc:
                images = page.get_images(full=True)
                image_count += len(images)
        logger.info(f"{path} PDF has {image_count} image(s).")
        return image_count > 0
    except Exception as e:
        logger.error(f"Failed to process {path}: {e}")
        return False
