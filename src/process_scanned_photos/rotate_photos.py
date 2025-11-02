import argparse
import os
from pathlib import Path

from PIL import Image


def rotate_photos(
    input_dir: str, rotation_angle: int = 90, output_dir: str = None
) -> None:
    """
    Rotate all photos in the input directory and save them.

    Args:
        input_dir: Path to directory containing photos
        rotation_angle: Angle to rotate images (90, 180, 270). Default is 90.
        output_dir: Optional output directory. If None, overwrites original files.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}

    processed_count = 0

    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            try:
                # Open image
                with Image.open(file_path) as img:
                    # Rotate image
                    rotated_img = img.rotate(-rotation_angle, expand=True)

                    # Save to output location
                    output_file = output_path / file_path.name
                    rotated_img.save(output_file, quality=95)

                    processed_count += 1
                    print(f"Rotated: {file_path.name}")

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

    print(f"\nProcessed {processed_count} images")


def main():
    parser = argparse.ArgumentParser(description="Rotate all photos in a directory")
    parser.add_argument(
        "input_dir", type=str, help="Path to directory containing photos to rotate"
    )
    parser.add_argument(
        "-a",
        "--angle",
        type=int,
        default=90,
        choices=[90, 180, 270],
        help="Rotation angle in degrees (default: 90)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: overwrite original files)",
    )

    args = parser.parse_args()

    rotate_photos(args.input_dir, args.angle, args.output_dir)


if __name__ == "__main__":
    main()
