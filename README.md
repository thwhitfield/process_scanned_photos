# Process Journal Photos

This was created with chatGPT on 2025-10-31 with the name split images script. Here's the link: https://chatgpt.com/c/69056c57-bd80-832e-bdac-0102570bf892

## Quickstart
1. Take photos of journal, 2 pages at a time, then download them from google photos into a folder
2. Run `python split_scan_pages.py ./in ./out --mode vertical --gutter 10`
3. Run `python images_to_pdf.py ./out ./journal.pdf --scale 0.5 --jpeg-quality 70`
4. This seemed to work pretty well


split_pages.py

```
# Typical journal spread shot on a phone; vertical split; 10px seam gutter
python split_scan_pages.py ./in ./out --mode vertical --gutter 10

# If you shot a top/bottom spread, use horizontal split:
python split_scan_pages.py ./in ./out --mode horizontal

# Trim a bit less and add a bigger clean border
python split_scan_pages.py ./in ./out --trim 4 --pad 24
```

images_to_pdf.py

```
# Basic (no downscaling):
python images_to_pdf.py ./scans ./out/journal.pdf

# Shrink by quality only (smaller file, same pixel dimensions):
python images_to_pdf.py ./scans ./out/journal.pdf --jpeg-quality 70

# Shrink by resizing + quality (biggest impact):
# Half the pixels each direction (~¼ the data), good for viewing
python images_to_pdf.py ./scans ./out/journal_small.pdf --scale 0.5 --jpeg-quality 70

# Or cap page size, e.g., fit inside 2000px tall while preserving aspect
python images_to_pdf.py ./scans ./out/journal_small.pdf --max-height 2000 --jpeg-quality 70
```