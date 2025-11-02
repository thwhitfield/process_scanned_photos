#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


# ---------- helpers ----------
def _order_quad(pts):
    """Order 4 points (x,y) as tl, tr, br, bl."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _four_point_warp(image, quad):
    """Perspective warp so quad becomes a rectangle."""
    (tl, tr, br, bl) = _order_quad(quad)
    # target width/height = distances along top/bottom & left/right
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(round(max(widthA, widthB)))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(round(max(heightA, heightB)))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_CUBIC)
    return warped


def _detect_page_quad(bgr, debug=False):
    """Return a 4-pt quad around the page if found; else None."""
    h, w = bgr.shape[:2]
    scale = 1000.0 / max(h, w)
    if scale < 1.0:
        bgr_small = cv2.resize(
            bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        bgr_small = bgr.copy()

    gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
    # suppress noise and boost edges
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # adaptive threshold + Canny, then combine
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
    )
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    comb = cv2.bitwise_or(thr, edges)

    contours, _ = cv2.findContours(comb, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # pick largest reasonable quadrilateral-ish contour
    page_quad = None
    area_img = bgr_small.shape[0] * bgr_small.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:50]:
        area = cv2.contourArea(cnt)
        if area < 0.15 * area_img:  # ignore tiny
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            page_quad = approx.reshape(4, 2).astype(np.float32)
            break

    if page_quad is None:
        # fallback: use minAreaRect
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        page_quad = box.astype(np.float32)

    # scale back up to original coordinates
    inv_scale = 1.0 / scale
    page_quad *= inv_scale
    return page_quad


def _auto_trim(white_bgr, margin_px=0):
    """Trim uniform borders from a (already-warped) page; add margin."""
    # Convert to grayscale and find content bbox
    gray = cv2.cvtColor(white_bgr, cv2.COLOR_BGR2GRAY)
    # Binary inverse: content (darker ink/lines) becomes white
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Clean small specks
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    ys, xs = np.where(thr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return white_bgr  # nothing detected

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Apply margin, clipping to image bounds
    y0 = max(0, y0 - margin_px)
    x0 = max(0, x0 - margin_px)
    y1 = min(white_bgr.shape[0] - 1, y1 + margin_px)
    x1 = min(white_bgr.shape[1] - 1, x1 + margin_px)

    cropped = white_bgr[y0 : y1 + 1, x0 : x1 + 1]
    return cropped


def _ensure_min_margin(bgr, pad_px=0, bg_color=255):
    """Pad with white (or chosen) border to ensure clean frame."""
    if pad_px <= 0:
        return bgr
    h, w = bgr.shape[:2]
    return cv2.copyMakeBorder(
        bgr,
        pad_px,
        pad_px,
        pad_px,
        pad_px,
        borderType=cv2.BORDER_CONSTANT,
        value=[bg_color, bg_color, bg_color],
    )


def _deskew_small_angle(bgr):
    """Optional tiny deskew using text lines. Tries HoughLines; safe for near-upright pages."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return bgr
    angles = []
    for rho_theta in lines[:80]:
        rho, theta = rho_theta[0]
        # Convert to degrees and normalize near horizontal
        deg = theta * 180.0 / np.pi
        # Map to [-90, 90)
        deg = (deg + 90) % 180 - 90
        if -45 < deg < 45:
            angles.append(deg)
    if not angles:
        return bgr
    angle = np.median(angles)
    if abs(angle) < 0.4:  # don't over-rotate
        return bgr
    # rotate
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def straighten_and_crop_page(pil_img, margin_trim_px=8, pad_px=12, small_deskew=True):
    """
    1) Honor EXIF orientation.
    2) Detect page quadrilateral and perspective-warp to top-down.
    3) Auto-trim a bit of background; add a clean margin (white).
    4) Optional small-angle deskew (safe).
    Returns a PIL.Image (RGB).
    """
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    quad = _detect_page_quad(bgr)
    if quad is not None:
        warped = _four_point_warp(bgr, quad)
    else:
        warped = bgr  # fallback: use as-is

    trimmed = _auto_trim(warped, margin_px=margin_trim_px)
    if small_deskew:
        trimmed = _deskew_small_angle(trimmed)
    padded = _ensure_min_margin(trimmed, pad_px=pad_px, bg_color=255)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------- splitting ----------
def split_image_center(pil_img, mode="vertical", gutter_px=0):
    """
    Split at midpoint. Optional gutter removes pixels around the center seam.
    Returns two PIL images.
    """
    w, h = pil_img.size
    if mode == "vertical":
        mid = w // 2
        left_box = (0, 0, mid - gutter_px // 2, h)
        right_box = (mid + math.ceil(gutter_px / 2), 0, w, h)
        a = pil_img.crop(left_box)
        b = pil_img.crop(right_box)
    else:
        mid = h // 2
        top_box = (0, 0, w, mid - gutter_px // 2)
        bot_box = (0, mid + math.ceil(gutter_px / 2), w, h)
        a = pil_img.crop(top_box)
        b = pil_img.crop(bot_box)
    return a, b


def is_image_path(p: Path) -> bool:
    return p.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".heic",
        ".heif",
    }


def run_split_scan_pages(
    input_dir: Path,
    output_dir: Path,
    mode: str = "vertical",
    gutter: int = 0,
    trim: int = 8,
    pad: int = 12,
    overwrite: bool = False,
    suffixes: str = "L,R",
) -> int:
    """
    Process scanned images: detect, straighten, crop, and split.

    Args:
        input_dir: Folder with images to process
        output_dir: Folder to write split images
        mode: Split direction ("vertical" or "horizontal")
        gutter: Pixels removed around the center seam
        trim: Pixels to trim from background content (adaptive)
        pad: Pixels of clean white margin to add after trimming
        overwrite: Whether to overwrite existing outputs
        suffixes: Comma-separated suffixes for halves (e.g., "L,R")

    Returns:
        Total number of images written
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    left_suf, right_suf = (s.strip() for s in suffixes.split(",", 1))

    # Optional HEIC support
    try:
        import pillow_heif  # type: ignore

        pillow_heif.register_heif_opener()
    except Exception:
        pass

    # natural-ish filename sort
    def sort_key(p: Path):
        import re

        return [
            int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", p.name)
        ]

    images = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and is_image_path(p)],
        key=sort_key,
    )
    if not images:
        print("No images found.")
        return 0

    total = 0
    for img_path in images:
        try:
            with Image.open(img_path) as im:
                # 1) straighten + crop page
                fixed = straighten_and_crop_page(
                    im, margin_trim_px=trim, pad_px=pad, small_deskew=True
                )
                # 2) split
                a, b = split_image_center(fixed, mode=mode, gutter_px=gutter)

                stem = img_path.stem
                # Write HEIC/HEIF as JPEG for portability
                ext_in = img_path.suffix.lower()
                ext_out = ".jpg" if ext_in in {".heic", ".heif"} else ext_in

                out_a = output_dir / f"{stem}_{left_suf}{ext_out}"
                out_b = output_dir / f"{stem}_{right_suf}{ext_out}"

                if not overwrite and (out_a.exists() or out_b.exists()):
                    print(f"Skipping (exists): {img_path.name}")
                    continue

                save_kwargs = {}
                if ext_out in {".jpg", ".jpeg"}:
                    save_kwargs = {"quality": 95, "subsampling": 2, "optimize": True}

                a.save(out_a, **save_kwargs)
                b.save(out_b, **save_kwargs)
                print(f"Processed: {img_path.name} -> {out_a.name}, {out_b.name}")
                total += 2
        except Exception as e:
            print(f"ERROR: {img_path.name}: {e}")

    print(f"Done. Wrote {total} split images to {output_dir}")
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Detect, straighten, crop, and split journal images."
    )
    ap.add_argument("input_dir", type=Path, help="Folder with images")
    ap.add_argument("output_dir", type=Path, help="Folder to write split images")
    ap.add_argument(
        "--mode",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Split direction (default: vertical)",
    )
    ap.add_argument(
        "--gutter",
        type=int,
        default=0,
        help="Pixels removed around the center seam (default: 0)",
    )
    ap.add_argument(
        "--trim",
        type=int,
        default=8,
        help="Trim this many pixels of background content (adaptive) (default: 8)",
    )
    ap.add_argument(
        "--pad",
        type=int,
        default=12,
        help="Add this many pixels of clean white margin after trimming (default: 12)",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    ap.add_argument(
        "--suffixes", default="L,R", help="Suffixes for halves (default: L,R)"
    )
    args = ap.parse_args()

    run_split_scan_pages(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        gutter=args.gutter,
        trim=args.trim,
        pad=args.pad,
        overwrite=args.overwrite,
        suffixes=args.suffixes,
    )


if __name__ == "__main__":
    main()
