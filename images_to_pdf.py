#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def natural_sort_key(name: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", name)]


def load_image(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)  # fix iPhone rotations
    return im.convert("RGB")  # PDFs like RGB


def maybe_resize(
    im: Image.Image,
    scale: float = 1.0,
    max_w: int | None = None,
    max_h: int | None = None,
) -> Image.Image:
    w, h = im.size
    if scale and scale != 1.0:
        w = int(w * scale)
        h = int(h * scale)
    if max_w or max_h:
        # Fit inside box while preserving aspect
        mw = max_w or w
        mh = max_h or h
        ratio = min(mw / w, mh / h)
        if ratio < 1.0:
            w = int(w * ratio)
            h = int(h * ratio)
    if (w, h) != im.size:
        im = im.resize((max(1, w), max(1, h)), Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser(
        description="Combine a folder of images into a single PDF with optional downscaling."
    )
    ap.add_argument("input_dir", type=Path, help="Folder containing images")
    ap.add_argument(
        "output_pdf", type=Path, help="Path to output PDF (e.g., out/book.pdf)"
    )
    # Downsizing options
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale factor (e.g., 0.5 halves both width and height)",
    )
    ap.add_argument(
        "--max-width", type=int, default=None, help="Max width in pixels (fit inside)"
    )
    ap.add_argument(
        "--max-height", type=int, default=None, help="Max height in pixels (fit inside)"
    )
    # Compression / quality options
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for embedding (higher=better/bigger). Typical 60–90. Default 85.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Set a DPI hint in the PDF metadata (e.g., 150). Doesn't change pixels; use with scale/max-*.",
    )
    ap.add_argument("--progress", action="store_true", help="Print progress")
    args = ap.parse_args()

    # Optional HEIC/HEIF support
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:
        pass

    imgs = [p for p in args.input_dir.iterdir() if p.is_file() and is_image(p)]
    imgs.sort(key=lambda p: natural_sort_key(p.name))

    if not imgs:
        print("No images found.")
        return

    processed = []
    for i, p in enumerate(imgs, 1):
        try:
            im = load_image(p)
            im = maybe_resize(
                im, scale=args.scale, max_w=args.max_width, max_h=args.max_height
            )

            # To control PDF size, we can ensure each page is JPEG-compressed inside the PDF.
            # Pillow will embed images as-is if they’re RGB. We can pre-encode to JPEG bytes
            # with chosen quality by saving to a temporary in-memory JPEG and reopening.
            from io import BytesIO

            buf = BytesIO()
            im.save(
                buf,
                format="JPEG",
                quality=args.jpeg_quality,
                optimize=True,
                subsampling=2,
            )
            buf.seek(0)
            im_jpeg = Image.open(buf).convert("RGB")
            processed.append(im_jpeg)

            if args.progress:
                print(
                    f"[{i}/{len(imgs)}] {p.name} -> {im_jpeg.size[0]}x{im_jpeg.size[1]}"
                )
        except Exception as e:
            print(f"ERROR loading {p.name}: {e}")

    if not processed:
        print("No images processed.")
        return

    first, rest = processed[0], processed[1:]
    save_kwargs = {"save_all": True, "append_images": rest}
    if args.dpi:
        save_kwargs["resolution"] = args.dpi

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    first.save(args.output_pdf, format="PDF", **save_kwargs)

    total_px = sum(im.size[0] * im.size[1] for im in processed)
    print(
        f"Done. Wrote {args.output_pdf} with {len(processed)} pages. Total pixels: {total_px:,}"
    )


if __name__ == "__main__":
    main()
