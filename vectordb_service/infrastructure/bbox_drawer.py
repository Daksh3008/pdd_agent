"""Draw bounding boxes on frame images."""

import os
from PIL import Image, ImageDraw


def draw_bounding_box(
    image_path: str,
    bbox: list[float],
    output_dir: str,
    color: str = "red",
    width: int = 3,
) -> str:
    """Draw a bounding box on the image and save annotated copy.

    Args:
        image_path: Path to the source frame image.
        bbox: Normalised [x1, y1, x2, y2] coordinates (0.0–1.0).
        output_dir: Directory to save the annotated image.
        color: Box outline colour.
        width: Box outline width in pixels.

    Returns:
        Path to the saved annotated image.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_annotated.png")
    img.save(out_path)
    return out_path
