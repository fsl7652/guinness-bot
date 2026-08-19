"""
export_onnx.py

Converts trained PyTorch models to ONNX format for deployment.
Run after training all three models.

Usage:
    python training/export_onnx.py --models models --out ml/models
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision import models
import torch.nn as nn


IMAGE_SIZE = 224


def load_model(pt_path, meta_path, device):
    with open(meta_path) as f:
        meta = json.load(f)

    num_classes = meta['num_classes']
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.to(device).eval()
    return model, meta


def export(model, onnx_path, device):
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={'image': {0: 'batch'}},
        opset_version=12,
        dynamo=False,   # use legacy exporter, compatible with older opsets
    )
    print(f"  ✓ exported → {onnx_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', required=True, help='folder containing trained .pt files')
    parser.add_argument('--out',    required=True, help='output folder for .onnx files')
    args = parser.parse_args()

    models_dir = Path(args.models)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')  # export on CPU for portability

    exports = [
        ('glass',      models_dir / 'glass'  / 'glass_best.pt',
                       models_dir / 'glass'  / 'glass_meta.json',
                       out_dir / 'glass.onnx'),
        ('head_ratio', models_dir / 'pour'   / 'head_ratio_best.pt',
                       models_dir / 'pour'   / 'head_ratio_meta.json',
                       out_dir / 'head_ratio.onnx'),
        ('texture',    models_dir / 'pour'   / 'texture_best.pt',
                       models_dir / 'pour'   / 'texture_meta.json',
                       out_dir / 'texture.onnx'),
        ('colour_sep', models_dir / 'pour'   / 'colour_sep_best.pt',
                       models_dir / 'pour'   / 'colour_sep_meta.json',
                       out_dir / 'colour_sep.onnx'),
        ('splitg',     models_dir / 'splitg' / 'splitg_best.pt',
                       models_dir / 'splitg' / 'splitg_meta.json',
                       out_dir / 'splitg.onnx'),
    ]

    for name, pt_path, meta_path, onnx_path in exports:
        if not pt_path.exists():
            print(f"  [skip] {name} — {pt_path} not found")
            continue
        if not meta_path.exists():
            print(f"  [skip] {name} — {meta_path} not found")
            continue

        print(f"Exporting {name}...")
        model, meta = load_model(pt_path, meta_path, device)
        export(model, onnx_path, device)

        # Save class mapping alongside onnx
        class_map = {i: c for i, c in enumerate(meta['classes'])}
        with open(onnx_path.with_suffix('.json'), 'w') as f:
            json.dump({'classes': meta['classes'], 'class_map': class_map}, f, indent=2)

    print(f"\n✓ ONNX models ready in {out_dir}")
    print("Next: copy to ml/models/ and convert to TensorRT on the Nano")
    print("  trtexec --onnx=glass.onnx --saveEngine=glass.trt --fp16")


if __name__ == '__main__':
    main()