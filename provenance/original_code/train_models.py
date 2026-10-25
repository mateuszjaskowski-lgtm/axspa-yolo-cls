#!/usr/bin/env python3
"""New, deduplicated spine-morphology experiment: training only.

No function in the training entry point enumerates or opens the external test set.
Run on a Colab CUDA GPU; --output-dir should be on mounted Google Drive.
The protocol is prospective for this revision, not a reconstruction of old training.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
import spine_transforms
from spine_transforms import TRANSFORM_PROTOCOL, build_spine_transforms

CLASSES = ['normal', 'osteofity', 'parasyndesmofity', 'syndesmofity']
MODELS = ['yolo11l-cls', 'yolo26l-cls', 'resnet50', 'efficientnet_b0']
SEEDS = [0, 1, 2, 3, 4]
ULTRALYTICS_VERSION = '8.4.163'
SCHEMA_VERSION = 2
VALIDATION_BATCH_SIZE = 32
YOLO_CONFIG = dict(imgsz=640, batch=100, epochs=240, patience=240,
    optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, cos_lr=False, amp=True, workers=2, cache=False,
    fliplr=0.5, flipud=0.0, auto_augment=None, erasing=0.0,
    scale=0.0, dropout=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    deterministic=True, save=True, save_period=-1, plots=True, device=0,
    val=True, split='val', fraction=1.0, compile=False)
TV_CONFIG = dict(imgsz=640, resize_mode='whole_png_aspect_preserving_center_pad', batch=32, epochs=100,
    optimizer='SGD', lr=0.01, momentum=0.937, nesterov=True, weight_decay=0.0005,
    scheduler='CosineAnnealingLR', min_lr=0.0001, early_stopping_patience=20,
    workers=0, amp=True, horizontal_flip_p=0.5,
    color_jitter_brightness=0.1, color_jitter_contrast=0.1,
    randaugment=False, random_affine=False, random_erasing_p=0.0,
    normalization_mean=[0.485, 0.456, 0.406],
    normalization_std=[0.229, 0.224, 0.225],
    resnet50_weights='IMAGENET1K_V2', efficientnet_b0_weights='IMAGENET1K_V1')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def write_rows(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty table.')
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def package_versions():
    packages = ['torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow',
                'opencv-python', 'scipy', 'scikit-learn']
    out = {}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def environment():
    import torch
    return dict(created_utc=utcnow(), python=sys.version, packages=package_versions(),
        platform=sys.platform, torch_cuda=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        source_sha256=sha256(__file__), transform_source_sha256=sha256(spine_transforms.__file__), command=sys.argv,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def gpu_preflight():
    # Must be set before creating CUDA contexts.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import torchvision
    import ultralytics
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f'Expected ultralytics=={ULTRALYTICS_VERSION}, got {ultralytics.__version__}.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required. In Colab select Runtime > Change runtime type > GPU.')
    if torch.cuda.device_count() < 1:
        raise RuntimeError('No CUDA device found.')
    # A tiny forward operation catches incompatible Torch/Torchvision CUDA builds early.
    x = torch.randn(1, 3, 32, 32, device='cuda:0')
    layer = torch.nn.Conv2d(3, 2, 3).cuda()
    layer(x).sum().backward()
    del x, layer
    torch.cuda.empty_cache()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def image_records(split_dir):
    """Enumerate only the explicitly supplied split; RGB decoding matches models."""
    from PIL import Image
    split_dir = Path(split_dir)
    folders = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if folders != CLASSES:
        raise ValueError(f'Class folders must be {CLASSES}; got {folders} at {split_dir}.')
    rows = []
    for label, cls in enumerate(CLASSES):
        files = sorted((split_dir / cls).iterdir(), key=lambda p: p.name)
        for path in files:
            if path.is_dir():
                raise ValueError(f'Unexpected nested directory: {path}')
            if path.suffix.lower() != '.png':
                if path.name.startswith('.'):
                    continue
                raise ValueError(f'Only PNG input permitted: {path}')
            with Image.open(path) as im:
                im.load()
                rgb = im.convert('RGB')
                # Include dimensions: equal byte streams with different shapes are not duplicates.
                pixel = hashlib.sha256((str(rgb.size) + '|RGB|').encode() + rgb.tobytes()).hexdigest()
                width, height = im.size
            rows.append(dict(relative_path=f'{cls}/{path.name}', true_label=label,
                class_name=cls, sha256=sha256(path), rgb_sha256=pixel,
                width=width, height=height))
    return rows


def development_manifest(dataset_root):
    """Intentionally never touch dataset_root/test, even to count its files."""
    root = Path(dataset_root)
    rows = []
    for split, expected in [('train', 579), ('val', 95)]:
        items = image_records(root / split)
        if len(items) != expected:
            raise ValueError(f'{split}: expected {expected} files after deduplication, got {len(items)}.')
        rows.extend(dict(row, split=split, relative_path=f'{split}/{row["relative_path"]}') for row in items)
    hashes = {}
    for row in rows:
        for field in ['sha256', 'rgb_sha256']:
            key = (field, row[field])
            if key in hashes:
                raise ValueError(f'Duplicate image remains in development: {hashes[key]} and {row["relative_path"]}')
            hashes[key] = row['relative_path']
    return rows


def torchvision_transforms(training):
    """New whole-PNG protocol: common geometry/augmentation across all backbones."""
    return build_spine_transforms(training, 'torchvision', TV_CONFIG['imgsz'])


def make_torchvision_model(name, weights=None):
    import torch
    import torchvision.models as models
    if name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    else:
        raise ValueError(name)
    return model


def prepare_pretrained(name, output):
    """Download generic official ImageNet weights once; never accept study weights."""
    import torch
    import torchvision.models as models
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / (name + '.json')
    if metadata_path.exists():
        info = json.loads(metadata_path.read_text())
        path = output / info['filename']
        if not path.is_file() or sha256(path) != info['sha256']:
            raise RuntimeError('Previously recorded generic pretrained file has changed: ' + str(path))
        return info
    if name.startswith('yolo'):
        from ultralytics import YOLO
        from ultralytics.utils.downloads import safe_download
        # Use isolated temporary cwd so an unrelated locally named .pt cannot be selected.
        prior_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='imagenet_pretrained_') as directory:
            try:
                os.chdir(directory)
                asset_url = f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt'
                source = Path(directory) / (name + '.pt')
                safe_download(url=asset_url, file=source, min_bytes=100000, retry=3, unzip=False)
                model = YOLO(str(source), task='classify')
                if len(model.names) != 1000:
                    raise RuntimeError('Generic YOLO ImageNet weights must contain 1000 classes.')
                destination = output / (name + '_imagenet.pt')
                shutil.copy2(source, destination)
                info = dict(model=name, filename=destination.name, sha256=sha256(destination),
                    source=asset_url,
                    requested_asset=name + '.pt', ultralytics=ULTRALYTICS_VERSION,
                    pretrained_class_count=len(model.names),
                    original_architecture=str(getattr(model.model, 'yaml', {})),
                    downloaded_utc=utcnow())
                del model
            finally:
                os.chdir(prior_cwd)
    else:
        weights = (models.ResNet50_Weights.IMAGENET1K_V2 if name == 'resnet50'
                   else models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # torchvision checks the official filename hash prefix during download.
        weights.get_state_dict(progress=True, check_hash=True)
        source = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
        expected_hash_prefix = source.stem.rsplit('-', 1)[-1]
        if not sha256(source).startswith(expected_hash_prefix):
            raise RuntimeError('Cached torchvision generic weights fail official SHA256 prefix verification.')
        destination = output / (name + '_imagenet.pth')
        shutil.copy2(source, destination)
        info = dict(model=name, filename=destination.name, sha256=sha256(destination),
                    source=weights.url, weight_enum=str(weights), pretrained_class_count=1000,
                    downloaded_utc=utcnow())
    atomic_json(metadata_path, info)
    return info


def init_torchvision_from_recorded(name, pretrained_dir, pretrained_info):
    import torch
    import torchvision.models as models
    source = Path(pretrained_dir) / pretrained_info['filename']
    if sha256(source) != pretrained_info['sha256']:
        raise RuntimeError('Generic pretrained SHA-256 mismatch.')
    state = torch.load(source, map_location='cpu', weights_only=True)
    if name == 'resnet50':
        model = models.resnet50(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    else:
        model = models.efficientnet_b0(weights=None)
        model.load_state_dict(state, strict=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def get_rng_state(generator):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                loader_generator=generator.get_state())


def restore_rng_state(state, generator):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader_generator'])


def check_environment_for_resume(path, current):
    previous = json.loads(Path(path).read_text())
    keys = ['python', 'packages', 'torch_cuda', 'cudnn_version', 'gpu', 'source_sha256', 'transform_source_sha256']
    changes = [k for k in keys if previous[k] != current[k]]
    if changes:
        raise RuntimeError('Interrupted run environment differs in ' + ', '.join(changes) +
                           '. Restart the run in a new preserved attempt; do not silently resume.')


def train_torchvision(name, seed, dataset_root, run_dir, pretrained_dir, pretrained_info, plan_hash):
    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    seed_everything(seed)
    run_dir = Path(run_dir)
    cfg = TV_CONFIG
    current_environment = environment()
    if (run_dir / 'environment.json').exists():
        check_environment_for_resume(run_dir / 'environment.json', current_environment)
    else:
        atomic_json(run_dir / 'environment.json', current_environment)
    train_dataset = ImageFolder(str(Path(dataset_root) / 'train'), transform=torchvision_transforms(True))
    val_dataset = ImageFolder(str(Path(dataset_root) / 'val'), transform=torchvision_transforms(False))
    if train_dataset.classes != CLASSES or val_dataset.classes != CLASSES:
        raise ValueError('Unexpected ImageFolder class order.')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch'], shuffle=True,
        num_workers=cfg['workers'], pin_memory=True, worker_init_fn=seed_worker,
        generator=generator, persistent_workers=False, drop_last=False)
    # Separate generator prevents validation iteration from changing the training RNG stream.
    val_generator = torch.Generator().manual_seed(999999)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch'], shuffle=False,
        num_workers=0, pin_memory=True, generator=val_generator)
    model = init_torchvision_from_recorded(name, pretrained_dir, pretrained_info).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
        nesterov=cfg['nesterov'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['min_lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    last = run_dir / 'last.pt'
    rows, start, best, best_epoch, stale = [], 0, -1.0, 0, 0
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint['plan_hash'] != plan_hash or checkpoint['model'] != name or checkpoint['seed'] != seed:
            raise RuntimeError('Resume checkpoint does not belong to this exact new experiment.')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # PyTorch optimizer.load_state_dict maps state to parameter devices.
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        rows = checkpoint['history']
        start, best, best_epoch, stale = (checkpoint[k] for k in ['next_epoch', 'best', 'best_epoch', 'stale'])
        restore_rng_state(checkpoint['rng'], generator)
    atomic_json(run_dir / 'effective_training_setup.json', dict(config=cfg,
        train_transforms=repr(train_dataset.transform), val_transforms=repr(val_dataset.transform),
        optimizer_class=type(optimizer).__name__, pretrained=pretrained_info,
        parameter_count=sum(p.numel() for p in model.parameters()),
        selection='Maximum internal validation top-1; earliest epoch on ties.',
        restart_policy='Exact epoch-boundary states saved; identical package/GPU environment required.',
        resume_count_start_epoch=start, protocol_is_new=True))
    started = time.perf_counter()
    cumulative_epoch_seconds = sum(row.get('epoch_training_validation_seconds', 0.0) for row in rows)
    for epoch in range(start, cfg['epochs']):
        if stale >= cfg['early_stopping_patience']:
            break
        epoch_started = time.perf_counter()
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        epoch_lr = optimizer.param_groups[0]['lr']
        for images, labels in train_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=cfg['amp']):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss: stopping without fallback.')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
        model.eval()
        val_loss, correct, count = 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                # float32 validation makes selection match exported inference.
                logits = model(images)
                loss = criterion(logits, labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise RuntimeError('Non-finite validation output.')
                val_loss += float(loss) * len(labels)
                correct += int((logits.argmax(1) == labels).sum())
                count += len(labels)
        accuracy = correct / count
        if accuracy > best:
            best, best_epoch, stale = accuracy, epoch + 1, 0
            atomic_torch_save(run_dir / 'selected.pt', dict(state_dict=model.state_dict(),
                model=name, class_names=CLASSES, seed=seed, plan_hash=plan_hash,
                epoch_1based=epoch + 1, val_accuracy=accuracy, preprocess=TV_CONFIG))
        else:
            stale += 1
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_started
        cumulative_epoch_seconds += epoch_seconds
        row = dict(epoch=epoch + 1, train_loss=total_loss / total,
            train_accuracy=total_correct / total, val_loss=val_loss / count,
            val_accuracy=accuracy, learning_rate=epoch_lr,
            best_epoch=best_epoch, best_val_accuracy=best,
            elapsed_current_session_seconds=time.perf_counter() - started,
            epoch_training_validation_seconds=epoch_seconds,
            cumulative_epoch_training_validation_seconds=cumulative_epoch_seconds)
        rows.append(row)
        atomic_torch_save(last, dict(model=name, seed=seed, plan_hash=plan_hash,
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
            next_epoch=epoch + 1, best=best, best_epoch=best_epoch, stale=stale,
            history=rows, rng=get_rng_state(generator), class_names=CLASSES))
        write_rows(run_dir / 'results.csv', rows)
        atomic_json(run_dir / 'progress.json', dict(status='training', **row, updated_utc=utcnow()))
        print(f'{name} seed={seed} epoch={epoch+1} val={accuracy:.6f} best={best:.6f}', flush=True)
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    return dict(best_epoch_1based=best_epoch, val_accuracy=best, epochs_completed=len(rows),
                active_epoch_training_validation_seconds=cumulative_epoch_seconds,
                timing_definition='Sum of completed epoch training/validation/selection work; excludes pauses and epoch checkpoint I/O.',
                backend='torchvision', preprocess=dict(TV_CONFIG))


def create_development_view(dataset_root, target):
    """Ultralytics receives a directory containing train and val only."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        link = target / split
        source = (Path(dataset_root) / split).resolve()
        if link.is_symlink() and link.resolve() == source:
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError('Unexpected existing development path: ' + str(link))
        link.symlink_to(source, target_is_directory=True)
    return target


def train_yolo(name, seed, dataset_root, run_dir, pretrained_dir, pretrained_info, plan_hash):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
    from ultralytics.data.dataset import ClassificationDataset
    seed_everything(seed)
    run_dir = Path(run_dir)
    if (run_dir / 'training').exists():
        raise RuntimeError('Interrupted YOLO run exists. Use --restart-incomplete-yolo to preserve it and restart '
                           'from recorded generic ImageNet weights. Native resume is intentionally not used.')
    atomic_json(run_dir / 'environment.json', environment())
    source = Path(pretrained_dir) / pretrained_info['filename']
    if sha256(source) != pretrained_info['sha256']:
        raise RuntimeError('Generic pretrained SHA-256 mismatch.')
    best = dict(accuracy=-1.0, epoch=0)
    callback_log = []
    epoch_timer = {'started': None}

    class WholePNGDataset(ClassificationDataset):
        def __init__(self, root, args, augment=False, prefix='', names=None):
            super().__init__(root, args, augment=augment, prefix=prefix, names=names)
            self.torch_transforms = build_spine_transforms(augment, 'ultralytics', args.imgsz)

    class Float32WholePNGValidator(ClassificationValidator):
        def __call__(self, trainer=None, model=None, **kwargs):
            # Pinned BaseValidator reads trainer.amp to choose half inputs/autocast.
            # Temporarily disable only that validation choice; restore training AMP.
            if trainer is None:
                self.args.quantize = None
                return super().__call__(trainer=trainer, model=model, **kwargs)
            previous_amp = trainer.amp
            try:
                trainer.amp = False
                return super().__call__(trainer=trainer, model=model, **kwargs)
            finally:
                trainer.amp = previous_amp

        def build_dataset(self, img_path):
            return WholePNGDataset(img_path, self.args, augment=False,
                                   prefix=self.args.split, names=self.names)

    class StrictClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode='train', batch=None):
            return WholePNGDataset(img_path, self.args, augment=mode == 'train',
                                   prefix='train' if mode == 'train' else self.args.split,
                                   names=self.data['names'])

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            # BaseTrainer otherwise doubles training batch for validation (200).
            if mode == 'train':
                return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)
            from ultralytics.data import build_dataloader
            from ultralytics.utils.torch_utils import is_parallel
            dataset = self.build_dataset(dataset_path, mode)
            loader = build_dataloader(dataset, VALIDATION_BATCH_SIZE, 0, shuffle=False,
                                      rank=rank, drop_last=False, device=self.device)
            target = self.model.module if is_parallel(self.model) else self.model
            target.transforms = dataset.torch_transforms
            return loader

        def get_validator(self):
            return Float32WholePNGValidator(self.test_loader, self.save_dir,
                                            args=copy.copy(self.args), _callbacks=self.callbacks)

        def save_model(self):
            if any(not torch.isfinite(v).all() for v in self.ema.ema.state_dict().values() if v.is_floating_point()):
                raise RuntimeError('Non-finite EMA weights; refusing framework clamp/recovery.')
            return super().save_model()

        def _handle_nan_recovery(self, epoch):
            # This revision must stop instead of rolling back/altering the training procedure.
            values = list(self.metrics.values()) + [self.fitness]
            if any(not math.isfinite(float(v)) for v in values if v is not None):
                raise RuntimeError('Non-finite YOLO metrics; automatic recovery disabled.')
            if self.loss is not None and not torch.isfinite(self.loss).all():
                raise RuntimeError('Non-finite YOLO loss; automatic recovery disabled.')
            return False

        def final_eval(self):
            # Selected checkpoint below uses strict top-1 and earliest ties, independently of framework best.pt.
            # Re-evaluation is performed once by predict_split on selected.pt after training.
            return None

    def on_epoch_start(trainer):
        epoch_timer['started'] = time.perf_counter()
        # Pinned source checks >=3 before automatically reducing a CUDA-OOM batch.
        trainer._oom_retries = 3
        if trainer.batch_size != YOLO_CONFIG['batch']:
            raise RuntimeError('Unexpected automatic batch-size modification.')
        torch.use_deterministic_algorithms(True)

    def on_setup(trainer):
        if len(trainer.train_loader.dataset) != 579 or len(trainer.test_loader.dataset) != 95:
            raise RuntimeError('Unexpected development dataset size inside Ultralytics.')
        if [trainer.data['names'][i] for i in range(4)] != CLASSES:
            raise RuntimeError('Unexpected Ultralytics class mapping.')
        params = [{k:v for k,v in group.items() if k != 'params'} for group in trainer.optimizer.param_groups]
        atomic_json(run_dir / 'effective_training_setup.json', dict(
            args=vars(trainer.args), optimizer_class=type(trainer.optimizer).__name__,
            optimizer_parameter_groups=params,
            train_transforms=repr(trainer.train_loader.dataset.torch_transforms),
            val_transforms=repr(trainer.test_loader.dataset.torch_transforms),
            validation_precision='float32', validation_batch_size=VALIDATION_BATCH_SIZE,
            validation_shuffle=False,
            spatial_augmentation_protocol=TRANSFORM_PROTOCOL,
            pretrained=pretrained_info,
            parameter_count=sum(p.numel() for p in trainer.model.parameters()),
            selection='Maximum internal validation top-1; earliest epoch on ties.',
            automatic_oom_batch_reduction=False, automatic_nan_recovery=False,
            native_resume_allowed=False, protocol_is_new=True))

    def on_saved(trainer):
        # BaseValidator rounds the returned dictionary to5decimalplaces.
        # Read its original metric, then express top1 as an integer fraction.
        raw_accuracy = float(trainer.validator.metrics.top1)
        if not math.isfinite(raw_accuracy):
            raise RuntimeError('Non-finite validation top-1.')
        val_count = len(trainer.test_loader.dataset)
        val_correct = int(round(raw_accuracy * val_count))
        accuracy = val_correct / val_count
        if abs(raw_accuracy - accuracy) > 1e-7:
            raise RuntimeError('Validation top1 is inconsistent with its sample count.')
        epoch = trainer.epoch + 1
        if accuracy > best['accuracy']:
            best.update(accuracy=accuracy, epoch=epoch)
            selected_model = copy.deepcopy(trainer.ema.ema).float().cpu()
            if hasattr(selected_model, 'criterion'):
                selected_model.criterion = None
            if any(not torch.isfinite(v).all() for v in selected_model.state_dict().values() if v.is_floating_point()):
                raise RuntimeError('Selected model contains non-finite weights.')
            atomic_torch_save(run_dir / 'selected.pt', dict(model=selected_model, ema=None,
                epoch=trainer.epoch, train_args=vars(trainer.args),
                train_metrics=trainer.metrics, version=ULTRALYTICS_VERSION,
                date=utcnow(), plan_hash=plan_hash, seed=seed,
                selection='strict_validation_top1_earliest_tie'))
            del selected_model
        row = dict(epoch=epoch, val_accuracy=accuracy, val_correct=val_correct, val_count=val_count,
                   framework_raw_top1=raw_accuracy, best_val_accuracy=best['accuracy'],
                   best_epoch=best['epoch'],
                   epoch_training_validation_seconds=time.perf_counter() - epoch_timer['started'])
        callback_log.append(row)
        write_rows(run_dir / 'selection_history.csv', callback_log)
        atomic_json(run_dir / 'progress.json', dict(status='training', **row, updated_utc=utcnow()))
        atomic_json(run_dir / 'last_checkpoint.json', dict(epoch_1based=epoch,
            filename=str(Path(trainer.last).relative_to(run_dir)),
            sha256=sha256(trainer.last), resume_supported=False,
            note='Preserved for audit; restart incomplete YOLO runs from generic pretrained weights.'))

    with tempfile.TemporaryDirectory(prefix='axspa_development_only_') as temp:
        development = create_development_view(dataset_root, Path(temp) / 'development')
        model = YOLO(str(source), task='classify')
        model.add_callback('on_pretrain_routine_end', on_setup)
        model.add_callback('on_train_epoch_start', on_epoch_start)
        model.add_callback('on_model_save', on_saved)
        model.train(trainer=StrictClassificationTrainer,
                    **YOLO_CONFIG, seed=seed, data=str(development),
                    project=str(run_dir), name='training', exist_ok=False,
                    pretrained=True, resume=False)
        results = run_dir / 'training' / 'results.csv'
        if results.exists():
            shutil.copy2(results, run_dir / 'results.csv')
        del model
    torch.cuda.empty_cache()
    if len(callback_log) != YOLO_CONFIG['epochs'] or best['epoch'] == 0:
        raise RuntimeError('YOLO run did not complete all prespecified revision epochs.')
    return dict(best_epoch_1based=best['epoch'], val_accuracy=best['accuracy'],
                epochs_completed=len(callback_log), backend='ultralytics',
                active_epoch_training_validation_seconds=sum(row['epoch_training_validation_seconds'] for row in callback_log),
                timing_definition='Sum of completed epoch work through model-save callback; includes validation/checkpoint save, excludes pauses.',
                preprocess=dict(imgsz=YOLO_CONFIG['imgsz'], geometry='whole_png_aspect_preserving_center_pad',
                    padding_rgb=[0,0,0], interpolation='PIL bilinear',
                    scale='0..1', mean=[0,0,0], std=[1,1,1],
                    source='stored SpineResizePad transforms; new revision protocol'))


def load_for_inference(run_dir, device='cuda:0'):
    """Load selected checkpoint only; return (model, transform, backend)."""
    import torch
    run_dir = Path(run_dir)
    info_path = run_dir / 'run_complete.json'
    if not info_path.exists():
        info_path = run_dir / 'selection_metadata.json'
    info = json.loads(info_path.read_text())
    checkpoint = run_dir / info['checkpoint']
    if sha256(checkpoint) != info['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint hash mismatch.')
    if info['backend'] == 'ultralytics':
        from ultralytics import YOLO
        model = YOLO(str(checkpoint), task='classify')
        if [model.names[i] for i in range(4)] != CLASSES:
            raise RuntimeError('Selected YOLO checkpoint has wrong class mapping.')
        attach_unfused_yolo_predictor(model, device)
        return model, getattr(model.model, 'transforms', None), 'ultralytics'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['class_names'] != CLASSES:
        raise RuntimeError('Selected torchvision checkpoint has wrong class mapping.')
    model = make_torchvision_model(saved['model'], weights=None)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.to(device).eval()
    return model, torchvision_transforms(False), 'torchvision'


def attach_unfused_yolo_predictor(yolo, device):
    """Keep export inference numerically aligned with FP32 unfused validation.

    Ultralytics' default predictor fuses Conv/BatchNorm; that algebraically
    equivalent operation can change a near-tied float32 class decision. This
    predictor is shared by the evaluator and Eigen-CAM via load_for_inference.
    """
    from ultralytics.models.yolo.classify import ClassificationPredictor
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.torch_utils import select_device

    class UnfusedClassificationPredictor(ClassificationPredictor):
        def setup_model(self, model, verbose=True):
            self.model = AutoBackend(model=copy.deepcopy(model),
                device=select_device(self.args.device, verbose=verbose),
                dnn=False, data=self.args.data, fp16=False,
                channels_last=False, fuse=False, verbose=verbose,
                end2end=self.args.nms is False)
            self.device = self.model.device
            self.model.eval()

    yolo.predictor = UnfusedClassificationPredictor(overrides=dict(
        task='classify', mode='predict', imgsz=YOLO_CONFIG['imgsz'], device=device,
        batch=VALIDATION_BATCH_SIZE, quantize=None, augment=False, save=False,
        verbose=False, compile=False), _callbacks=yolo.callbacks)
    yolo.predictor.setup_model(yolo.model, verbose=False)


def _discarded_amp_steps(model, optimizer, scaler, x, y, name,
                         max_attempts=32, target_updates=2):
    """Exercise real optimizer updates, allowing bounded AMP scale calibration.

    Preflight only: every parameter update is discarded. With the non-fused
    optimizers used here, GradScaler calls optimizer.step only for finite
    gradients, so the public post-step hook counts actual updates, not attempts.
    Study training uses its own freshly initialized model/optimizer/scaler.
    """
    import torch
    if any(group.get('fused', False) for group in optimizer.param_groups):
        raise ValueError('AMP preflight step counting requires a non-fused optimizer.')
    updates = 0
    trace = []

    def count_update(_optimizer, _args, _kwargs):
        nonlocal updates
        updates += 1

    handle = optimizer.register_step_post_hook(count_update)
    try:
        for attempt in range(1, max_attempts + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(x.device.type, enabled=scaler.is_enabled()):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name}: non-finite hardware preflight loss at attempt {attempt}.')
            scale_before = scaler.get_scale()
            updates_before = updates
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            executed = updates > updates_before
            trace.append(dict(attempt=attempt, optimizer_step_executed=executed,
                              scale_before=scale_before, scale_after=scaler.get_scale()))
            print(f'Hardware preflight {name}: attempt {attempt}/{max_attempts}, '
                  f'optimizer updates {updates}/{target_updates}, '
                  f'AMP scale {scale_before:g} -> {scaler.get_scale():g}'
                  + ('' if executed else ' (overflow: update skipped)'), flush=True)
            if updates == target_updates:
                break
    finally:
        handle.remove()
    if updates != target_updates or not optimizer.state:
        raise RuntimeError(f'{name}: hardware preflight did not complete {target_updates} '
                           f'optimizer updates in {len(trace)} attempts; actual updates={updates}, '
                           f'final AMP scale={scaler.get_scale():g}. '
                           'Preserve the log and investigate numerical stability; study training has not started.')
    return dict(discarded_optimizer_steps=updates, amp_attempts=len(trace),
                amp_skipped_steps=len(trace) - updates, amp_scale_trace=trace)


def hardware_preflight(dataset_root, output, pretrained_dir, pretrained, plan):
    """Exercise all real architectures/batches on training images before a study run.

    Two actual discarded optimizer steps allocate optimizer state, allowing up
    to 32 attempts for initial AMP scale calibration. YOLO also holds an
    EMA-sized copy. This tests hardware feasibility, not model performance.
    FP32 validation-shaped inference uses TRAIN images only. No study weights
    or validation/test performance are created, selected, or retained.
    """
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel
    current = environment()
    environment_key = canonical_hash({k: current[k] for k in
        ['python', 'packages', 'torch_cuda', 'cudnn_version', 'gpu', 'gpu_total_bytes',
         'source_sha256', 'transform_source_sha256']})
    report_path = Path(output) / 'hardware_preflight.json'
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if (prior.get('protocol_sha256') == plan['protocol_sha256'] and
                prior.get('environment_key') == environment_key and
                prior.get('status') == 'passed' and
                [row['model'] for row in prior.get('models', [])] == MODELS):
            print('All-model hardware preflight already passed for this protocol/environment.', flush=True)
            return prior
    rows = image_records(Path(dataset_root) / 'train')
    results = []
    for name in MODELS:
        seed_everything(987654)
        cfg = YOLO_CONFIG if name.startswith('yolo') else TV_CONFIG
        backend = 'ultralytics' if name.startswith('yolo') else 'torchvision'
        transform = build_spine_transforms(True, backend, cfg['imgsz'])
        eval_transform = build_spine_transforms(False, backend, cfg['imgsz'])
        images, targets = [], []
        for row in rows[:cfg['batch']]:
            with Image.open(Path(dataset_root) / 'train' / row['relative_path']) as image:
                images.append(transform(image.convert('RGB')))
            targets.append(row['true_label'])
        if name.startswith('yolo'):
            model = YOLO(str(Path(pretrained_dir) / pretrained[name]['filename']), task='classify').model
            ClassificationModel.reshape_outputs(model, len(CLASSES))
        else:
            model = init_torchvision_from_recorded(name, pretrained_dir, pretrained[name])
        model = model.float().cuda().train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        ema_sized_copy = copy.deepcopy(model).eval() if name.startswith('yolo') else None
        optimizer = (torch.optim.AdamW(model.parameters(), lr=0.001) if name.startswith('yolo')
                     else torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], nesterov=True))
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        x = torch.stack(images).cuda()
        y = torch.tensor(targets, device='cuda', dtype=torch.long)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        amp_probe = _discarded_amp_steps(model, optimizer, scaler, x, y, name)
        del x, y, images, targets
        model.eval()
        images = []
        for row in rows[:VALIDATION_BATCH_SIZE]:
            with Image.open(Path(dataset_root) / 'train' / row['relative_path']) as image:
                images.append(eval_transform(image.convert('RGB')))
        with torch.inference_mode():
            output_tensor = model(torch.stack(images).cuda().float())
            outputs = output_tensor if isinstance(output_tensor, (tuple, list)) else [output_tensor]
            if any(not torch.isfinite(tensor).all() for tensor in outputs):
                raise RuntimeError(f'{name}: non-finite FP32 preflight output.')
        torch.cuda.synchronize()
        results.append(dict(model=name, training_batch=cfg['batch'], imgsz=cfg['imgsz'],
            **amp_probe, float32_validation_batch=VALIDATION_BATCH_SIZE,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.perf_counter() - started, status='passed',
            no_study_weights_saved=True))
        del model, ema_sized_copy, optimizer, scaler, images, output_tensor, outputs
        torch.cuda.empty_cache()
        print(f'Hardware preflight passed: {name}, batch{cfg["batch"]}, {cfg["imgsz"]}px.', flush=True)
    report = dict(status='passed', created_utc=utcnow(), protocol_sha256=plan['protocol_sha256'],
        environment_key=environment_key, environment=current, models=results,
        data_split_used='train only', external_test_access=False,
        limitation='Two optimizer steps and FP32 inference test feasibility; not a guarantee against later resource exhaustion.',
        experimental_run_weights_reused=False)
    atomic_json(report_path, report)
    return report


def predict_split(run_dir, split_dir, batch_size=32, device='cuda:0'):
    """Explicit inference helper for named split. Training calls this on val only."""
    import torch
    import numpy as np
    from PIL import Image
    rows = image_records(split_dir)
    model, transform, backend = load_for_inference(run_dir, device=device)
    probabilities = []
    if backend == 'ultralytics':
        # The checked checkpoint contains the exact inference transform used in training.
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            files = [str(Path(split_dir) / row['relative_path']) for row in part]
            inference_size = getattr(transform.transforms[0], 'size', YOLO_CONFIG['imgsz'])
            predictions = model.predict(source=files, imgsz=inference_size, batch=batch_size,
                device=device, half=False, augment=False, verbose=False, stream=False, save=False)
            if len(predictions) != len(files):
                raise RuntimeError('YOLO prediction count mismatch.')
            for expected, prediction in zip(files, predictions):
                if Path(prediction.path).resolve() != Path(expected).resolve():
                    raise RuntimeError('YOLO result path/order mismatch.')
                probabilities.append(prediction.probs.data.detach().cpu().float().numpy())
    else:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                part = rows[start:start + batch_size]
                images = []
                for row in part:
                    with Image.open(Path(split_dir) / row['relative_path']) as image:
                        images.append(transform(image.convert('RGB')))
                logits = model(torch.stack(images).to(device))
                probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
    array = np.asarray(probabilities, dtype=float)
    if array.shape != (len(rows), 4) or not np.isfinite(array).all():
        raise RuntimeError('Invalid probability output shape or non-finite values.')
    if not np.allclose(array.sum(axis=1), 1, atol=1e-5) or np.any(array < 0) or np.any(array > 1):
        raise RuntimeError('Invalid class probability vectors.')
    for row, vector in zip(rows, array):
        row['probabilities'] = vector.tolist()
        row['predicted_label'] = int(vector.argmax())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def create_or_check_plan(output, manifest, clean_manifest_path):
    output = Path(output)
    path = output / 'training_plan.json'
    protocol = dict(schema_version=SCHEMA_VERSION, class_names=CLASSES, classes=CLASSES,
        models=MODELS, seeds=SEEDS, reference_seed=0,
        expected_runs=[dict(model=m, seed=s, relative_run_dir=f'{m}/seed_{s}') for m in MODELS for s in SEEDS],
        development_manifest_sha256=canonical_hash(manifest),
        development_manifest_hash_definition='canonical JSON of train/val records only',
        clean_manifest_sha256=sha256(clean_manifest_path),
        clean_manifest_hash_definition='SHA256 of full clean_manifest.csv bytes; no test images opened',
        configs=dict(ultralytics=YOLO_CONFIG, torchvision=TV_CONFIG),
        runner_sha256=sha256(__file__), ultralytics_version=ULTRALYTICS_VERSION,
        transform_source_sha256=sha256(spine_transforms.__file__),
        spatial_augmentation_protocol=TRANSFORM_PROTOCOL,
        validation_precision='float32', validation_batch_size=VALIDATION_BATCH_SIZE,
        yolo_inference_conv_batchnorm_fusion=False,
        external_test_previously_examined=True,
        primary_model='yolo11l-cls',
        experiment='New cleaned-data revision experiment; not historical training reconstruction.',
        checkpoint_selection='Maximum val top-1 with earliest epoch on ties, separately for each model/seed.',
        primary_seed_selection='Fixed reference seed0; no external or validation selection among seeds.',
        comparison_scope='Training pipelines at640px; common whole-PNG geometry/augmentation; optimizers, normalization and epoch budgets differ.',
        training_timing_comparable_across_backends=False,
        timing_limitation='Epoch timing scopes differ in checkpoint I/O; record operational timings, do not rank architecture efficiency from them.',
        alpha_handling='Discard PNG alpha channel without compositing; retain stored RGB values.',
        patient_identity_independence='Requires author/local linkage; image hashes alone cannot verify patients.',
        torchvision_config_provenance='Optimizer/budget based on recovered notebook; common whole-PNG transforms newly specified before retraining.',
        deliberate_changes_from_old_torchvision='Five seeds, whole-PNG aspect-preserving resize+pad, mild common augmentation, explicit RNG streams, float32 validation/inference, locked test, hash audit.',
        external_test_access_during_training=False,
        expected_train=579, expected_val=95, expected_external=150)
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['protocol_sha256'] != canonical_hash(protocol):
            raise RuntimeError('Existing frozen plan differs. Create a new experiment output directory.')
    else:
        plan = dict(protocol, protocol_sha256=canonical_hash(protocol), created_utc=utcnow())
        atomic_json(path, plan)
        atomic_json(output / 'development_manifest.json', manifest)
        write_rows(output / 'development_manifest.csv', manifest)
        shutil.copy2(__file__, output / 'train_models_frozen.py')
        shutil.copy2(spine_transforms.__file__, output / 'spine_transforms.py')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS,
                        help='Run subset of frozen plan now; plan always contains all four models.')
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=SEEDS,
                        help='Run subset now; evaluation remains blocked until all planned runs complete.')
    parser.add_argument('--restart-incomplete-yolo', action='store_true',
                        help='Preserve interrupted YOLO attempt and restart same seed from generic pretrained weights.')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate development input/freeze plan without GPU or training.')
    parser.add_argument('--hardware-preflight-only', action='store_true',
                        help='Test all four actual architectures/full batches on TRAIN images; discard weights and stop.')
    args = parser.parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'external_evaluation_started.json').exists():
        raise RuntimeError('External evaluation has started for this frozen experiment. Further training is locked.')
    clean_manifest_path = args.data_root.parent / 'clean_manifest.csv'
    if not clean_manifest_path.is_file():
        raise FileNotFoundError('Run prepare_data.py first: clean_manifest.csv is required next to dataset/.')
    manifest = development_manifest(args.data_root)
    plan = create_or_check_plan(args.output_dir, manifest, clean_manifest_path)
    if args.validate_only:
        print(json.dumps(dict(status='development_validated', train=579, val=95,
                              test_opened=False, protocol_sha256=plan['protocol_sha256']), indent=2))
        return
    gpu_preflight()
    # Keep an environment record and package lock before downloading/training.
    invocation = args.output_dir / 'invocations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    invocation.mkdir(parents=True, exist_ok=True)
    atomic_json(invocation / 'environment.json', environment())
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (invocation / 'requirements_freeze.txt').write_text(freeze.stdout)
    pretrained_dir = args.output_dir / 'generic_imagenet_weights'
    pretrained = {name: prepare_pretrained(name, pretrained_dir) for name in MODELS}
    hardware_preflight(args.data_root, args.output_dir, pretrained_dir, pretrained, plan)
    if args.hardware_preflight_only:
        print('All hardware checks passed. No study training run has started.', flush=True)
        return
    for model in args.models:
        for seed in args.seeds:
            run_dir = args.output_dir / model / f'seed_{seed}'
            complete = run_dir / 'run_complete.json'
            if complete.exists():
                info = json.loads(complete.read_text())
                checkpoint = run_dir / info['checkpoint']
                if info['protocol_sha256'] != plan['protocol_sha256'] or sha256(checkpoint) != info['checkpoint_sha256']:
                    raise RuntimeError('Completed run metadata/checkpoint changed.')
                print(f'Skipping completed {model} seed{seed}', flush=True)
                continue
            selection_path = run_dir / 'selection_metadata.json'
            pending_finalization = None
            if selection_path.exists():
                candidate = json.loads(selection_path.read_text())
                if (candidate.get('protocol_sha256') != plan['protocol_sha256'] or
                        candidate.get('model') != model or candidate.get('seed') != seed or
                        sha256(run_dir / candidate['checkpoint']) != candidate['checkpoint_sha256']):
                    raise RuntimeError('Pending completed-training metadata/checkpoint changed.')
                if model.startswith('yolo') and candidate.get('epochs_completed') != YOLO_CONFIG['epochs']:
                    raise RuntimeError('Pending YOLO finalization lacks the full epoch budget.')
                pending_finalization = candidate
                print(f'Recovering validation/export only for completed training {model} seed{seed}.', flush=True)
            if (pending_finalization is None and model.startswith('yolo') and
                    (run_dir / 'training').exists() and args.restart_incomplete_yolo):
                archive = args.output_dir / 'interrupted_attempts' / model / f'seed_{seed}_{time.time_ns()}'
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(run_dir), str(archive))
            run_dir.mkdir(parents=True, exist_ok=True)
            started_record = dict(model=model, seed=seed,
                protocol_sha256=plan['protocol_sha256'], started_utc=utcnow(),
                pretrained=pretrained[model], status='finalizing' if pending_finalization else 'training',
                external_test_access=False)
            if not (run_dir / 'run_started.json').exists():
                atomic_json(run_dir / 'run_started.json', started_record)
            atomic_json(run_dir / 'run_invocations' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.json'),
                        started_record)
            try:
                if pending_finalization is not None:
                    info = metadata = pending_finalization
                else:
                    trainer = train_yolo if model.startswith('yolo') else train_torchvision
                    metadata = trainer(model, seed, args.data_root, run_dir,
                                       pretrained_dir, pretrained[model], plan['protocol_sha256'])
                    info = dict(metadata, model=model, seed=seed, checkpoint='selected.pt',
                        checkpoint_sha256=sha256(run_dir / 'selected.pt'),
                        protocol_sha256=plan['protocol_sha256'], class_names=CLASSES,
                        pretrained=pretrained[model], external_test_access=False)
                    atomic_json(selection_path, info)
                predictions = predict_split(run_dir, args.data_root / 'val')
                reproduced = sum(row['predicted_label'] == row['true_label'] for row in predictions) / len(predictions)
                atomic_json(run_dir / 'val_predictions.json', predictions)
                flattened = [dict(relative_path='val/' + row['relative_path'],
                    true_label=row['true_label'], predicted_label=row['predicted_label'],
                    sha256=row['sha256'], **{f'p_{c}':row['probabilities'][i] for i,c in enumerate(CLASSES)})
                    for row in predictions]
                write_rows(run_dir / 'val_predictions.csv', flattened)
                # Stop on an altered top1 after serialization/normal inference; investigate instead of hiding it.
                if abs(reproduced - metadata['val_accuracy']) > 1e-7:
                    raise RuntimeError(f'Selected checkpoint val accuracy differs: training {metadata["val_accuracy"]}, '
                                       f'reloaded {reproduced}. Preserve run for review.')
                atomic_json(complete, dict(info, status='complete', completed_utc=utcnow(),
                    val_accuracy_reproduced=reproduced,
                    val_predictions_sha256=sha256(run_dir / 'val_predictions.json')))
            except BaseException as exc:
                atomic_json(run_dir / 'run_failed.json', dict(status='incomplete', failed_utc=utcnow(),
                    error_type=type(exc).__name__, message=str(exc), external_test_access=False))
                raise
    expected = plan['expected_runs']
    completed = [r for r in expected if (args.output_dir / r['relative_run_dir'] / 'run_complete.json').exists()]
    atomic_json(args.output_dir / 'training_status.json', dict(expected_runs=len(expected),
        completed_runs=len(completed), all_training_complete=len(completed) == len(expected),
        external_test_access=False, updated_utc=utcnow()))
    print(f'Training complete for requested subset. Finished {len(completed)}/{len(expected)} frozen runs. '
          'No external test images were opened.', flush=True)


if __name__ == '__main__':
    main()
