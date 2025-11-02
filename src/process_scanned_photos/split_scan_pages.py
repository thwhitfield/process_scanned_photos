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


def _deskew_small_angle(bgr):
    """Deskew by detecting dominant near-vertical structures via multiple Hough variants."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 160)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        9,
    )
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 45))
    seam_enhanced = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
    combined = cv2.bitwise_or(edges, seam_enhanced)

    angles: list[float] = []
    weights: list[float] = []

    primary = _collect_hough_angles(combined)
    angles.extend(primary)
    weights.extend([1.0] * len(primary))

    if len(angles) < 5:
        refined_edges = cv2.Canny(seam_enhanced, 30, 120)
        secondary = _collect_hough_angles(refined_edges, max_lines=240)
        angles.extend(secondary)
        weights.extend([0.8] * len(secondary))

    prob_angles, prob_weights = _collect_houghp_angles(combined)
    angles.extend(prob_angles)
    weights.extend(prob_weights)

    if not angles:
        return bgr

    angles_np = np.array(angles, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)
    median = np.median(angles_np)
    mask = np.abs(angles_np - median) <= 6.0
    if mask.sum() >= 3:
        angles_np = angles_np[mask]
        weights_np = weights_np[mask]

    total_weight = float(weights_np.sum())
    if total_weight <= 1e-6:
        angle = float(np.median(angles_np))
    else:
        angle = float(np.sum(angles_np * weights_np) / total_weight)

    angle = float(np.clip(angle, -10.0, 10.0))
    if abs(angle) < 0.15:  # negligible
        return bgr

    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Min-max normalize 1D profile safely."""
    prof = profile.astype(np.float32)
    prof -= prof.min()
    max_val = prof.max()
    if max_val > 1e-6:
        prof /= max_val
    else:
        prof[:] = 0.0
    return prof


def _theta_to_vertical(theta_rad: float) -> float:
    """Convert Hough theta (normal angle) to deviation from vertical in degrees."""
    deg = theta_rad * 180.0 / np.pi
    return (deg + 90.0) % 180.0 - 90.0


def _collect_hough_angles(edge_map: np.ndarray, max_lines: int = 180) -> list[float]:
    """Return orientation samples (degrees) from standard Hough transform."""
    angles: list[float] = []
    threshold = max(100, int(0.18 * min(edge_map.shape[:2])))
    lines = cv2.HoughLines(edge_map, 1, np.pi / 180.0, threshold=threshold)
    if lines is None:
        return angles
    for rho_theta in lines[:max_lines]:
        theta = float(rho_theta[0][1])
        angle = _theta_to_vertical(theta)
        if abs(angle) <= 20.0:
            angles.append(angle)
    return angles


def _collect_houghp_angles(edge_map: np.ndarray) -> tuple[list[float], list[float]]:
    """Return orientation samples and weights from probabilistic Hough transform."""
    angles: list[float] = []
    weights: list[float] = []
    h, w = edge_map.shape[:2]
    min_len = max(40, int(0.15 * min(h, w)))
    max_gap = max(20, int(0.05 * min(h, w)))
    lines = cv2.HoughLinesP(
        edge_map,
        1,
        np.pi / 180.0,
        threshold=max(60, int(0.04 * min(h, w))),
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if lines is None:
        return angles, weights
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = (angle + 90.0) % 180.0 - 90.0
        if abs(angle) <= 20.0:
            length = math.hypot(x2 - x1, y2 - y1)
            angles.append(angle)
            weights.append(max(length, 1.0))
    return angles, weights


def _prepare_image_for_split(pil_img, apply_deskew=True):
    """Return an RGB PIL image ready for seam detection (EXIF orientation + optional deskew)."""
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    if not apply_deskew:
        return pil_img

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    aligned = _deskew_small_angle(bgr)
    if aligned is bgr:
        return pil_img
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _find_split_position(gray, mode="vertical"):
    """Heuristically locate the spine seam in the grayscale image."""
    working = gray if mode == "vertical" else gray.T
    blur = cv2.GaussianBlur(working, (5, 5), 0)
    h, w = blur.shape

    # Projection profiles
    column_mean = blur.mean(axis=0).astype(np.float32)
    edges = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    edge_profile = np.mean(np.abs(edges), axis=0)

    # Contrast between left/right windows around each candidate seam.
    idxs = np.arange(w, dtype=np.int32)
    window = max(15, int(round(w * 0.04)))
    if window >= w:
        window = max(1, w // 2)
    cumsum = np.concatenate(([0.0], np.cumsum(column_mean, dtype=np.float64)))

    left_start = np.maximum(0, idxs - window)
    left_sum = cumsum[idxs] - cumsum[left_start]
    left_count = np.maximum(1, idxs - left_start)
    left_mean = left_sum / left_count

    right_end = np.minimum(w, idxs + window)
    right_sum = cumsum[right_end] - cumsum[idxs]
    right_count = np.maximum(1, right_end - idxs)
    right_mean = right_sum / right_count

    contrast = np.abs(left_mean - right_mean)

    # Normalize feature profiles and lightly smooth to reduce noise.
    edge_norm = _normalize_profile(
        cv2.GaussianBlur(edge_profile.reshape(1, -1), (1, 9), 0).ravel()
    )
    contrast_norm = _normalize_profile(
        cv2.GaussianBlur(contrast.reshape(1, -1), (1, 9), 0).ravel()
    )
    dark_norm = _normalize_profile(255.0 - column_mean)

    # Favor central columns to avoid selecting outer edges.
    center = (w - 1) / 2.0
    sigma = max(8.0, w * 0.17)
    center_weight = np.exp(-((idxs - center) ** 2) / (2 * sigma * sigma)).astype(
        np.float32
    )

    score = center_weight * (
        0.6 * edge_norm + 0.3 * contrast_norm + 0.1 * dark_norm
    )

    start = max(0, int(round(w * 0.18)))
    end = min(w, int(round(w * 0.82)))
    if end - start < 15:
        start, end = 0, w

    window_scores = score[start:end]
    if window_scores.size == 0 or np.ptp(window_scores) < 1e-3:
        seam = int(round(center))
    else:
        local_idx = int(np.argmax(window_scores))
        seam = start + local_idx

    seam = int(np.clip(seam, 0, w - 1))
    return seam if mode == "vertical" else seam


# ---------- splitting ----------
def split_image_center(
    pil_img,
    mode="vertical",
    gutter_px=0,
    overlap_px=0,
    overlap_frac=0.1,
    auto_seam=True,
):
    """
    Split the image using a detected seam (or midpoint fallback).
    - gutter removes pixels around the center seam.
    - overlap keeps the central area on both halves (positive values undo gutter).
    Returns two PIL images.
    """
    w, h = pil_img.size
    gutter_px = max(0, int(gutter_px))
    overlap_px = max(0, int(overlap_px))
    overlap_frac = max(0.0, float(overlap_frac))

    if auto_seam:
        gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        seam = _find_split_position(gray, mode=mode)
    else:
        seam = w // 2 if mode == "vertical" else h // 2

    if mode == "vertical":
        base_overlap = int(round(overlap_frac * w))
        overlap_total = max(overlap_px, base_overlap)

        half_left = seam - gutter_px // 2
        half_right = seam + math.ceil(gutter_px / 2)

        left_end = min(w, max(0, half_left) + overlap_total)
        right_start = max(0, min(w, half_right) - overlap_total)

        left_end = max(1, min(w, left_end))
        right_start = max(0, min(w - 1, right_start))
        if right_start > left_end:
            mid = (right_start + left_end) // 2
            left_end = max(1, mid)
            right_start = max(0, mid)

        left_box = (0, 0, left_end, h)
        right_box = (right_start, 0, w, h)
        a = pil_img.crop(left_box)
        b = pil_img.crop(right_box)
    else:
        base_overlap = int(round(overlap_frac * h))
        overlap_total = max(overlap_px, base_overlap)

        half_top = seam - gutter_px // 2
        half_bottom = seam + math.ceil(gutter_px / 2)

        top_end = min(h, max(0, half_top) + overlap_total)
        bottom_start = max(0, min(h, half_bottom) - overlap_total)

        top_end = max(1, min(h, top_end))
        bottom_start = max(0, min(h - 1, bottom_start))
        if bottom_start > top_end:
            mid = (bottom_start + top_end) // 2
            top_end = max(1, mid)
            bottom_start = max(0, mid)

        top_box = (0, 0, w, top_end)
        bot_box = (0, bottom_start, w, h)
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
    overlap: int = 0,
    overlap_frac: float = 0.1,
    deskew: bool = True,
    overwrite: bool = False,
    suffixes: str = "L,R",
) -> int:
    """
    Process scanned images: optionally deskew, locate the seam, and split.

    Args:
        input_dir: Folder with images to process
        output_dir: Folder to write split images
        mode: Split direction ("vertical" or "horizontal")
        gutter: Pixels removed around the center seam
        overlap: Pixels of overlap to retain around the seam when splitting
        overlap_frac: Fraction of page width/height to overlap (each side retains this much)
        deskew: Whether to attempt a gentle deskew before splitting
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
                prepared = _prepare_image_for_split(
                    im,
                    apply_deskew=deskew,
                )
                # split using detected seam
                a, b = split_image_center(
                    prepared,
                    mode=mode,
                    gutter_px=gutter,
                    overlap_px=overlap,
                    overlap_frac=overlap_frac,
                    auto_seam=True,
                )

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
        description="Detect the seam in scanned spreads and split them into individual pages."
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
        "--overlap",
        type=int,
        default=0,
        help="Additional pixels of overlap around the seam (default: 0)",
    )
    ap.add_argument(
        "--overlap-frac",
        type=float,
        default=0.1,
        help="Fraction of page width/height to keep as overlap on both halves (default: 0.1)",
    )
    ap.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable the small-angle deskew before seam detection",
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
        overlap=args.overlap,
        overlap_frac=args.overlap_frac,
        deskew=not args.no_deskew,
        overwrite=args.overwrite,
        suffixes=args.suffixes,
    )


if __name__ == "__main__":
    main()
