from pathlib import Path
from tempfile import TemporaryDirectory

from process_scanned_photos.images_to_pdf import convert_images_to_pdf
from process_scanned_photos.split_scan_pages import run_split_scan_pages


def process_journal_pipeline(
    input_dir: Path,
    output_pdf: Path,
    mode: str = "vertical",
    gutter: int = 0,
    trim: int = 8,
    pad: int = 12,
    suffixes: str = "L,R",
    scale: float = 1.0,
    max_width: int | None = None,
    max_height: int | None = None,
    jpeg_quality: int = 85,
    dpi: int | None = None,
    progress: bool = False,
    keep_split_images: bool = False,
    split_output_dir: Path | None = None,
) -> None:
    """
    Complete pipeline: split scanned journal pages, then combine into a PDF.

    Args:
        input_dir: Folder containing scanned images
        output_pdf: Path to output PDF file
        mode: Split direction ("vertical" or "horizontal")
        gutter: Pixels removed around the center seam
        trim: Pixels to trim from background content (adaptive)
        pad: Pixels of clean white margin to add after trimming
        suffixes: Comma-separated suffixes for halves (e.g., "L,R")
        scale: Uniform scale factor for PDF images
        max_width: Max width in pixels for PDF images
        max_height: Max height in pixels for PDF images
        jpeg_quality: JPEG quality for PDF embedding (60-90 typical)
        dpi: DPI hint in PDF metadata
        progress: Print progress messages
        keep_split_images: If True, keep split images in split_output_dir
        split_output_dir: Where to save split images (if keep_split_images=True)
    """
    if keep_split_images:
        if split_output_dir is None:
            raise ValueError(
                "split_output_dir must be provided if keep_split_images=True"
            )
        temp_dir = split_output_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        use_temp = False
    else:
        # Use a temporary directory that will be cleaned up
        temp_dir_obj = TemporaryDirectory()
        temp_dir = Path(temp_dir_obj.name)
        use_temp = True

    try:
        # Step 1: Split scanned pages
        if progress:
            print(f"Step 1: Splitting scanned pages from {input_dir}...")

        num_split = run_split_scan_pages(
            input_dir=input_dir,
            output_dir=temp_dir,
            mode=mode,
            gutter=gutter,
            trim=trim,
            pad=pad,
            overwrite=True,
            suffixes=suffixes,
        )

        if num_split == 0:
            print("No images were split. Aborting.")
            return

        # Step 2: Convert split images to PDF
        if progress:
            print(f"\nStep 2: Converting {num_split} split images to PDF...")

        convert_images_to_pdf(
            input_dir=temp_dir,
            output_pdf=output_pdf,
            scale=scale,
            max_width=max_width,
            max_height=max_height,
            jpeg_quality=jpeg_quality,
            dpi=dpi,
            progress=progress,
        )

        print(f"\n✓ Pipeline complete! Output: {output_pdf}")
        if keep_split_images:
            print(f"✓ Split images saved in: {temp_dir}")

    finally:
        # Clean up temporary directory if used
        if use_temp:
            temp_dir_obj.cleanup()


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Complete pipeline: split scanned journal pages and convert to PDF."
    )
    ap.add_argument("input_dir", type=Path, help="Folder containing scanned images")
    ap.add_argument("output_pdf", type=Path, help="Path to output PDF file")

    # Split options
    split_group = ap.add_argument_group("split options")
    split_group.add_argument(
        "--mode",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Split direction (default: vertical)",
    )
    split_group.add_argument(
        "--gutter", type=int, default=0, help="Pixels removed around center seam"
    )
    split_group.add_argument(
        "--trim", type=int, default=8, help="Pixels to trim from background"
    )
    split_group.add_argument(
        "--pad", type=int, default=12, help="Pixels of white margin to add"
    )
    split_group.add_argument(
        "--suffixes", default="L,R", help="Suffixes for split halves"
    )

    # PDF options
    pdf_group = ap.add_argument_group("PDF options")
    pdf_group.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor for PDF images"
    )
    pdf_group.add_argument("--max-width", type=int, help="Max width in pixels")
    pdf_group.add_argument("--max-height", type=int, help="Max height in pixels")
    pdf_group.add_argument(
        "--jpeg-quality", type=int, default=85, help="JPEG quality (60-90 typical)"
    )
    pdf_group.add_argument("--dpi", type=int, help="DPI hint for PDF metadata")

    # Output options
    ap.add_argument("--progress", action="store_true", help="Print progress messages")
    ap.add_argument(
        "--keep-split-images",
        action="store_true",
        help="Keep split images instead of deleting them",
    )
    ap.add_argument(
        "--split-output-dir",
        type=Path,
        help="Where to save split images (required if --keep-split-images)",
    )

    args = ap.parse_args()

    process_journal_pipeline(
        input_dir=args.input_dir,
        output_pdf=args.output_pdf,
        mode=args.mode,
        gutter=args.gutter,
        trim=args.trim,
        pad=args.pad,
        suffixes=args.suffixes,
        scale=args.scale,
        max_width=args.max_width,
        max_height=args.max_height,
        jpeg_quality=args.jpeg_quality,
        dpi=args.dpi,
        progress=args.progress,
        keep_split_images=args.keep_split_images,
        split_output_dir=args.split_output_dir,
    )


if __name__ == "__main__":
    main()
