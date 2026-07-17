"""
batch_segment.py

Runs all images in an input folder through segment.py and saves
each detected glass crop to an output folder.

Also writes a manifest.json mapping each crop back to its source photo,
which Label Studio can use and which you'll need when training.

Usage:
    python batch_segment.py <input_dir> <output_dir> [--debug]

Example:
    python batch_segment.py data/raw data/crops --debug

Output structure:
    data/crops/
        img_001_glass1.jpg
        img_001_glass2.jpg
        img_002_glass1.jpg
        ...
        manifest.json
"""

import cv2
import json
import sys
import argparse
from pathlib import Path


SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp'}


def batch_segment(input_dir, output_dir, debug=False):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in SUPPORTED
    ])

    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images — running segmentation...")

    # Lazy import so script fails clearly if deps missing
    from segment import get_glass_crops

    manifest    = []
    total_crops = 0
    failed      = 0

    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img_path.name}", end="  ")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print("✗ could not load")
            failed += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            crops = get_glass_crops(img_rgb, debug=debug)
        except Exception as e:
            print(f"✗ segmentation failed: {e}")
            failed += 1
            continue

        if not crops:
            print("0 glasses")
            # Save the original as a no-glass example — useful for not_glass class
            out_name = f"{img_path.stem}_noglass.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), img_bgr)
            manifest.append({
                "crop_file":   out_name,
                "source_file": img_path.name,
                "glass_index": 0,
                "sam_score":   None,
                "bbox":        None,
                "hint":        "no_glass_detected",
            })
            continue

        print(f"{len(crops)} glass(es)")

        for g in crops:
            out_name = f"{img_path.stem}_glass{g['index']+1}.jpg"
            out_path = output_dir / out_name

            crop_bgr = cv2.cvtColor(g["crop"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), crop_bgr)

            manifest.append({
                "crop_file":   out_name,
                "source_file": img_path.name,
                "glass_index": g["index"] + 1,
                "sam_score":   round(g["score"], 4),
                "bbox":        list(g["bbox"]),
                "hint":        None,   # filled in by Label Studio export
            })
            total_crops += 1

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'─'*40}")
    print(f"Done.")
    print(f"  Images processed: {len(images) - failed}/{len(images)}")
    print(f"  Crops saved:      {total_crops}")
    print(f"  Failed:           {failed}")
    print(f"  Manifest:         {manifest_path}")
    print(f"\nNext step: import {output_dir}/*.jpg into Label Studio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch segment pint photos into glass crops")
    parser.add_argument("input_dir",  help="Folder containing raw pint photos")
    parser.add_argument("output_dir", help="Folder to save crops and manifest")
    parser.add_argument("--debug",    action="store_true", help="Print SAM debug info")
    args = parser.parse_args()

    batch_segment(args.input_dir, args.output_dir, debug=args.debug)