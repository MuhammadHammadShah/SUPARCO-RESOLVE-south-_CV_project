from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int):

    return A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5,
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2(),
    ])


def get_valid_transforms(image_size: int):

    return A.Compose([
        A.Resize(image_size, image_size),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2(),
    ])