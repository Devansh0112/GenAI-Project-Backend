import base64
import io
import re
from PIL import Image


def base64_to_image(base64_string):
    """Convert a base64 string to a PIL Image"""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def image_to_base64(image):
    """Convert a PIL Image to a base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def clean_prompt(text):
    """Clean prompt text for better results"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text