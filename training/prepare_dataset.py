"""
prepare_dataset.py

Converts Label Studio JSON-MIN export + batch_segment manifest into
organised train/val/test folders ready for PyTorch ImageFolder.

Usage:
    python training/prepare_dataset.py --labels data/labels.json --manifest data/crops/manifest.json --crops data/crops --out data/datasets
"""

import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from urllib.parse import unquote


TRAIN = 0.75
VAL   = 0.15
TEST  = 0.10
SEED  = 42


def extract_filename(image_url):
    """
    Extract just the filename from a Label Studio image URL.
    Handles paths like: /data/local-files/?d=mnt/c/Users/.../crops/file.jpg
    """
    decoded = unquote(image_url)
    # Strip the ?d= query param prefix and get everything after it
    if '?d=' in decoded:
        path_part = decoded.split('?d=')[1]
    else:
        path_part = decoded
    # Get just the filename
    return Path(path_part).name


def parse_label_studio_export(labels_path):
    with open(labels_path, encoding='utf-8') as f:
        tasks = json.load(f)

    parsed = {}
    for task in tasks:
        filename = extract_filename(task.get('image', ''))
        if not filename:
            continue

        labels = {
            'glass_type':     task.get('glass_type'),
            'head_ratio':     task.get('head_ratio'),
            'texture':        task.get('head_texture'),   # Label Studio exports as head_texture
            'colour_sep':     task.get('colour_sep'),
            'splitg_quality': task.get('splitg_quality'),
        }

        if labels['glass_type']:
            parsed[filename] = labels

    return parsed


def split_files(files, seed=SEED):
    random.seed(seed)
    files = list(files)
    random.shuffle(files)
    n       = len(files)
    n_train = int(n * TRAIN)
    n_val   = int(n * VAL)
    return {
        'train': files[:n_train],
        'val':   files[n_train:n_train + n_val],
        'test':  files[n_train + n_val:],
    }


def copy_files(file_list, class_name, split_name, dest_root, crops_dir):
    dest = dest_root / split_name / class_name
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for filename in file_list:
        src = crops_dir / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
            copied += 1
        else:
            print(f"  [warn] crop not found: {filename}")
    return copied


def build_glass_classifier(labels, crops_dir, out_dir):
    print("\n── Glass classifier ─────────────────────────")
    dest     = out_dir / 'glass_classifier'
    by_class = defaultdict(list)

    for filename, ann in labels.items():
        by_class[ann['glass_type']].append(filename)

    summary = {}
    for class_name, files in by_class.items():
        splits = split_files(files)
        print(f"  {class_name}: {len(files)} total "
              f"({len(splits['train'])} train / "
              f"{len(splits['val'])} val / "
              f"{len(splits['test'])} test)")
        for split_name, split_files_ in splits.items():
            copy_files(split_files_, class_name, split_name, dest, crops_dir)
        summary[class_name] = {s: len(f) for s, f in splits.items()}

    return summary


def build_pour_quality(labels, crops_dir, out_dir):
    print("\n── Pour quality ─────────────────────────────")
    dest   = out_dir / 'pour_quality'
    tulips = {f: ann for f, ann in labels.items()
              if ann['glass_type'] == 'guinness_tulip'}

    summary = {}
    for label_name in ['head_ratio', 'texture', 'colour_sep']:
        by_class = defaultdict(list)
        for filename, ann in tulips.items():
            value = ann.get(label_name)
            if value:
                by_class[value].append(filename)

        label_dest    = dest / label_name
        label_summary = {}
        print(f"\n  {label_name}:")
        for class_name, files in by_class.items():
            splits = split_files(files)
            print(f"    {class_name}: {len(files)} "
                  f"({len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])})")
            for split_name, split_files_ in splits.items():
                copy_files(split_files_, class_name, split_name, label_dest, crops_dir)
            label_summary[class_name] = {s: len(f) for s, f in splits.items()}

        summary[label_name] = label_summary

    return summary


def build_splitg(labels, crops_dir, out_dir):
    print("\n── Split-the-G ──────────────────────────────")
    dest     = out_dir / 'splitg'
    by_class = defaultdict(list)

    for filename, ann in labels.items():
        if ann['glass_type'] == 'guinness_midsip':
            value = ann.get('splitg_quality')
            if value:
                by_class[value].append(filename)

    summary = {}
    for class_name, files in by_class.items():
        splits = split_files(files)
        print(f"  {class_name}: {len(files)} "
              f"({len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])})")
        for split_name, split_files_ in splits.items():
            copy_files(split_files_, class_name, split_name, dest, crops_dir)
        summary[class_name] = {s: len(f) for s, f in splits.items()}

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels',   required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--crops',    required=True)
    parser.add_argument('--out',      required=True)
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from {args.labels}...")
    labels = parse_label_studio_export(args.labels)
    print(f"  {len(labels)} annotated crops found")

    # Debug: show first few filenames found
    if labels:
        for i, fn in enumerate(list(labels.keys())[:3]):
            print(f"  sample: {fn} → {labels[fn]['glass_type']}")
    else:
        print("  [debug] checking raw image URLs...")
        with open(args.labels, encoding='utf-8') as f:
            tasks = json.load(f)
        for task in tasks[:3]:
            url = task.get('image', '')
            fn  = extract_filename(url)
            print(f"  url: {url[:80]}...")
            print(f"  → filename: {fn}")
            print(f"  → glass_type: {task.get('glass_type')}")

    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    manifest_files = {m['crop_file'] for m in manifest}
    missing = set(labels.keys()) - manifest_files
    if missing:
        print(f"  [warn] {len(missing)} labelled files not in manifest")

    glass_summary  = build_glass_classifier(labels, crops_dir, out_dir)
    pour_summary   = build_pour_quality(labels, crops_dir, out_dir)
    splitg_summary = build_splitg(labels, crops_dir, out_dir)

    summary = {
        'total_labelled':   len(labels),
        'glass_classifier': glass_summary,
        'pour_quality':     pour_summary,
        'splitg':           splitg_summary,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Done — datasets written to {out_dir}")
    print(f"✓ {len(labels)} images organised into train/val/test splits")


if __name__ == '__main__':
    main()