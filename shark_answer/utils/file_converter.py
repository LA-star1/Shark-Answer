"""File format conversion utilities for Shark Answer.

Converts uploaded files (PDF, DOCX, HEIC, WEBP) into
lists of PNG image bytes ready for the AI vision extractor.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions that are directly usable as images (no conversion needed)
_NATIVE_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# Extensions requiring Pillow conversion
_PILLOW_IMAGE_EXTS = {".webp", ".gif", ".bmp", ".tiff", ".tif"}
# HEIC/HEIF needs pillow-heif
_HEIC_EXTS = {".heic", ".heif"}


def convert_file_to_images(filename: str, data: bytes) -> list[bytes]:
    """Convert any supported file to a list of PNG image bytes.

    Supported formats:
    - JPG / JPEG / PNG  → returned as-is (no re-encode needed)
    - WEBP / BMP / TIFF → converted to PNG via Pillow
    - HEIC / HEIF       → converted via pillow-heif
    - PDF               → each page rendered to PNG via PyMuPDF
    - DOCX / DOC        → text rendered to PNG + embedded images extracted

    Returns an empty list on error (caller should decide whether to skip or abort).
    """
    ext = Path(filename).suffix.lower()

    if ext in _NATIVE_IMAGE_EXTS:
        return [data]
    if ext in _PILLOW_IMAGE_EXTS:
        return _convert_pillow_image(data)
    if ext in _HEIC_EXTS:
        return _convert_heic(data)
    if ext == ".pdf":
        return _convert_pdf(data)
    if ext in {".docx", ".doc"}:
        return _convert_docx(data)

    # Unknown extension: try to pass through as an image
    logger.warning("Unknown file extension '%s'; treating as raw image bytes.", ext)
    return [data]


# ---------------------------------------------------------------------------
# Private converters
# ---------------------------------------------------------------------------

def _convert_pillow_image(data: bytes) -> list[bytes]:
    """Convert WEBP / GIF / BMP / TIFF to PNG using Pillow."""
    try:
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(data))
        img = _ensure_rgb(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return [buf.getvalue()]
    except ImportError:
        logger.error("Pillow is not installed; cannot convert image format.")
        return [data]  # return original and hope for the best
    except Exception as exc:
        logger.error("Pillow image conversion failed: %s", exc)
        return [data]


def _convert_heic(data: bytes) -> list[bytes]:
    """Convert HEIC / HEIF to PNG using pillow-heif + Pillow."""
    try:
        import pillow_heif  # type: ignore
        from PIL import Image  # type: ignore

        pillow_heif.register_heif_opener()
        img = Image.open(io.BytesIO(data))
        img = _ensure_rgb(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        logger.info("Converted HEIC/HEIF image to PNG.")
        return [buf.getvalue()]
    except ImportError:
        logger.error(
            "pillow-heif is not installed. Run: pip install pillow-heif"
        )
        return []
    except Exception as exc:
        logger.error("HEIC conversion failed: %s", exc)
        return []


def _convert_pdf(data: bytes) -> list[bytes]:
    """Render each PDF page to a PNG image at 150 DPI using PyMuPDF."""
    try:
        import fitz  # type: ignore  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        images: list[bytes] = []
        # 2× scale ≈ 144 DPI — good balance between quality and file size
        matrix = fitz.Matrix(2.0, 2.0)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))
        doc.close()
        logger.info("Converted PDF to %d page image(s).", len(images))
        return images
    except ImportError:
        logger.error(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        )
        return []
    except Exception as exc:
        logger.error("PDF conversion failed: %s", exc)
        return []


def _convert_docx(data: bytes) -> list[bytes]:
    """Extract text + embedded images from a DOCX/DOC file.

    Returns a list where the first element (if any text exists) is a rendered
    PNG of the paragraph text, followed by any embedded images found in the doc.
    """
    text_paragraphs: list[str] = []
    embedded_images: list[bytes] = []

    try:
        from docx import Document as DocxDocument  # type: ignore

        doc = DocxDocument(io.BytesIO(data))

        # Collect paragraph text
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if stripped:
                text_paragraphs.append(stripped)

        # Collect embedded images from relationships
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    embedded_images.append(rel.target_part.blob)
                except Exception:
                    pass

    except ImportError:
        logger.error(
            "python-docx is not installed. Run: pip install python-docx"
        )
        return []
    except Exception as exc:
        logger.error("DOCX extraction failed: %s", exc)
        return []

    results: list[bytes] = []

    # Render text to a PNG image so the vision model can read it
    if text_paragraphs:
        text_content = "\n".join(text_paragraphs)
        text_png = _text_to_png(text_content)
        if text_png:
            results.append(text_png)

    results.extend(embedded_images)
    logger.info(
        "Extracted %d text page(s) + %d embedded image(s) from DOCX.",
        1 if text_paragraphs else 0,
        len(embedded_images),
    )
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_rgb(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """Convert image to RGB if needed (drops alpha for JPEG compat)."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _text_to_png(text: str, max_chars: int = 6000) -> bytes | None:
    """Render plain text to a white-background PNG using Pillow.

    Falls back to None if Pillow isn't available or rendering fails.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(content truncated)"

        FONT_SIZE = 18
        MARGIN = 40
        LINE_HEIGHT = FONT_SIZE + 8
        IMG_WIDTH = 1200

        # Try to find a decent monospace font
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        for font_path in (
            "/System/Library/Fonts/Courier.dfont",
            "/Library/Fonts/Courier New.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        ):
            try:
                font = ImageFont.truetype(font_path, FONT_SIZE)
                break
            except Exception:
                continue
        else:
            font = ImageFont.load_default()

        # Wrap lines to fit image width
        chars_per_line = max(1, int((IMG_WIDTH - 2 * MARGIN) / (FONT_SIZE * 0.55)))
        wrapped: list[str] = []
        for raw in text.split("\n"):
            if not raw:
                wrapped.append("")
                continue
            while len(raw) > chars_per_line:
                wrapped.append(raw[:chars_per_line])
                raw = raw[chars_per_line:]
            wrapped.append(raw)

        height = min(MARGIN * 2 + len(wrapped) * LINE_HEIGHT, 5000)

        img = Image.new("RGB", (IMG_WIDTH, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = MARGIN
        for line in wrapped:
            if y + LINE_HEIGHT > height - MARGIN:
                # Draw a "more content..." indicator
                draw.text((MARGIN, y), "...(page clipped)", fill=(150, 150, 150), font=font)
                break
            draw.text((MARGIN, y), line, fill=(20, 20, 20), font=font)
            y += LINE_HEIGHT

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except ImportError:
        logger.warning("Pillow not installed; cannot render DOCX text as image.")
        return None
    except Exception as exc:
        logger.error("Text-to-PNG rendering failed: %s", exc)
        return None
