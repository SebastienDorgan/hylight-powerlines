"""Image I/O and transforms for the LLM pipeline."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image

from .models import Box


def ensure_dir(p: Path) -> None:
    """Create `p` if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> Image.Image:
    """Read an image from disk and convert it to RGB."""
    return Image.open(path).convert("RGB")


def img_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    """Encode an image as JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def img_to_b64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """Encode an image as base64 JPEG."""
    return base64.b64encode(img_to_jpeg_bytes(img, quality=quality)).decode("ascii")


def crop_with_box(img: Image.Image, b: Box) -> tuple[Image.Image, tuple[int, int]]:
    """Crop an image using a box (pixel coords), returning crop and origin offset."""
    import math

    x1 = math.floor(b.x1)
    y1 = math.floor(b.y1)
    x2 = math.ceil(b.x2)
    y2 = math.ceil(b.y2)
    crop = img.crop((x1, y1, x2, y2))
    return crop, (x1, y1)
