"""Whole-PNG transforms shared by all four spine classification pipelines.

This module must remain beside the training/evaluation scripts when loading
checkpoints: SpineResizePad is serialized by its stable module/class name.
"""
from __future__ import annotations

TRANSFORM_PROTOCOL = {
    'geometry': 'Preserve the entire PNG; resize longest edge, preserve aspect ratio, center-pad to square.',
    'size': 640,
    'padding_rgb': [0, 0, 0],
    'interpolation': 'PIL bilinear',
    'alpha_handling': 'Discard alpha without compositing; retain stored RGB values.',
    'training_horizontal_flip_p': 0.5,
    'training_brightness': 0.1,
    'training_contrast': 0.1,
    'random_crop': False,
    'random_affine': False,
    'randaugment': False,
    'random_erasing': False,
    'yolo_normalization': {'mean': [0, 0, 0], 'std': [1, 1, 1]},
    'torchvision_normalization': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
}


class SpineResizePad:
    """Resize without removing pixels or changing anatomical aspect ratio.

    Integer `.size` also allows Ultralytics' classification predictor to select
    this saved transform instead of reverting to its default center crop.
    """

    def __init__(self, size=640):
        if not isinstance(size, int) or size <= 0:
            raise ValueError('size must be a positive integer')
        self.size = size

    def geometry(self, image_size):
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions must be positive.')
        scale = self.size / max(width, height)
        resized_width = max(1, min(self.size, round(width * scale)))
        resized_height = max(1, min(self.size, round(height * scale)))
        return resized_width, resized_height, (self.size - resized_width) // 2, (self.size - resized_height) // 2

    def __call__(self, image):
        from PIL import Image
        image = image.convert('RGB')
        width, height, left, top = self.geometry(image.size)
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        canvas.paste(resized, (left, top))
        return canvas

    def __repr__(self):
        return f'SpineResizePad(size={self.size}, interpolation=BILINEAR, fill=(0,0,0), crop=False)'


def build_spine_transforms(training, backend, size=640):
    """Shared spatial/intensity policy; keep the backbone-specific normalization.

    Brightness/contrast augment the source image before resizing/padding so padding
    remains exactly black. These are online training transforms only.
    """
    from torchvision import transforms as T
    if backend not in {'ultralytics', 'torchvision'}:
        raise ValueError(backend)
    operations = []
    if training:
        operations += [T.RandomHorizontalFlip(TRANSFORM_PROTOCOL['training_horizontal_flip_p']),
                       T.ColorJitter(brightness=TRANSFORM_PROTOCOL['training_brightness'],
                                     contrast=TRANSFORM_PROTOCOL['training_contrast'])]
    # For validation this is first: Ultralytics uses its integer `.size` to
    # recognize the stored inference transform during prediction.
    operations += [SpineResizePad(size)]
    normalization = TRANSFORM_PROTOCOL['yolo_normalization' if backend == 'ultralytics' else 'torchvision_normalization']
    operations += [T.ToTensor(), T.Normalize(normalization['mean'], normalization['std'])]
    return T.Compose(operations)
