"""Deterministic, development-only visual and numeric input preflight.

Shows one training PNG per class, complete resized/padded field, and one seeded
online augmentation. Does not enumerate or open validation/external images.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spine_transforms import SpineResizePad, TRANSFORM_PROTOCOL, build_spine_transforms

CLASSES = ['normal', 'osteofity', 'parasyndesmofity', 'syndesmofity']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_rgb(tensor, backend):
    if tensor.shape != (3, 640, 640) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise AssertionError('Unexpected model input shape, dtype or non-finite value')
    key = 'yolo_normalization' if backend == 'ultralytics' else 'torchvision_normalization'
    mean = torch.tensor(TRANSFORM_PROTOCOL[key]['mean'])[:, None, None]
    std = torch.tensor(TRANSFORM_PROTOCOL[key]['std'])[:, None, None]
    rgb = (tensor.cpu() * std + mean).permute(1, 2, 0).numpy()
    if rgb.min() < -1e-6 or rgb.max() > 1 + 1e-6:
        raise AssertionError('De-normalized model input is outside RGB range')
    return np.clip(rgb, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if sorted(p.name for p in args.train_root.iterdir() if p.is_dir()) != CLASSES:
        raise ValueError('Unexpected training class folders')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    resize = SpineResizePad(640)
    records = []
    for i, cls in enumerate(CLASSES):
        source = sorted((args.train_root / cls).glob('*.png'))[0]
        with Image.open(source) as original:
            original.load()
            image = original.convert('RGB')
        nw, nh, left, top = resize.geometry(image.size)
        if not (max(nw, nh) == 640 and min(nw, nh) > 0 and left >= 0 and top >= 0
                and left + nw <= 640 and top + nh <= 640):
            raise AssertionError('Invalid whole-field resize/pad geometry')
        scale = 640 / max(image.size)
        if abs(nw - image.width * scale) > 0.500001 or abs(nh - image.height * scale) > 0.500001:
            raise AssertionError('Geometry changes anatomical aspect ratio beyond integer rounding')
        expected = np.asarray(resize(image), dtype=np.float32) / 255
        seeded_outputs = {}
        for backend in ('ultralytics', 'torchvision'):
            evaluated = display_rgb(build_spine_transforms(False, backend)(image), backend)
            if not np.allclose(evaluated, expected, atol=1e-6, rtol=0):
                raise AssertionError('Evaluation pipeline does not preserve expected whole-field geometry')
            torch.manual_seed(1704 + i)
            augmented = display_rgb(build_spine_transforms(True, backend)(image), backend)
            mask = np.ones((640, 640), dtype=bool)
            mask[top:top + nh, left:left + nw] = False
            if np.any(np.abs(augmented[mask]) > 1e-6):
                raise AssertionError('Augmentation changed black padding')
            seeded_outputs[backend] = (evaluated, augmented)
        if any(not np.allclose(a, b, atol=1e-6, rtol=0)
               for a, b in zip(seeded_outputs['ultralytics'], seeded_outputs['torchvision'])):
            raise AssertionError('Backends differ spatially or in intensity before normalization')
        images = (np.asarray(image), expected, seeded_outputs['ultralytics'][1])
        titles = ('Received PNG / stored RGB', 'Whole field: resize + black padding', 'One training augmentation')
        for j, (visual, title) in enumerate(zip(images, titles)):
            axes[i, j].imshow(visual)
            axes[i, j].set_title(f'{cls} — {title}', fontsize=9)
            axes[i, j].axis('off')
        records.append(dict(class_name=cls, true_label=i, relative_path=f'train/{cls}/{source.name}',
            sha256=digest(source), source_size=list(image.size), resized_size=[nw, nh],
            padding_left_top=[left, top], output_size=[640, 640], source_crop_box=None,
            source_field_retained=True, augmentation_seed=1704 + i,
            dtype='float32', alpha_handling=TRANSFORM_PROTOCOL['alpha_handling']))
    fig.suptitle('Technical input preflight — training images only; not model results', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    target = args.output_dir / 'training_input_preview.png'
    fig.savefig(target, dpi=180, facecolor='white')
    plt.close(fig)
    with Image.open(target) as verified_preview:
        verified_preview.load()
    metadata = dict(status='passed', source_split='train', test_images_opened=False,
        selection='Lexicographically first PNG per training class; no outcome-based selection',
        transform_protocol=TRANSFORM_PROTOCOL, transform_module_sha256=digest(Path(__file__).with_name('spine_transforms.py')),
        preview_script_sha256=digest(__file__), preview_png_sha256=digest(target), images=records,
        checks=['all source spatial extent retained with aspect ratio and <=0.5px rounding',
                'equal spatial/intensity transforms across all models before backbone normalization',
                'finite float32 tensors, expected shape/range and unchanged black padding'])
    (args.output_dir / 'training_input_preview.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Input preflight passed: all four training examples retain their complete field; no test images opened.')
    print(target)


if __name__ == '__main__':
    main()
