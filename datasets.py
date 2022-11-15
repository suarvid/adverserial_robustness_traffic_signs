import torch
import albumentations as alb
import numpy as np
import os

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2

# Useful constants
IMAGE_DIR: str = "../input/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
VALID_SPLIT: float = 0.1
IMAGE_DIM: int = 224
BATCH_SIZE: int = 128
NUM_WORKERS: int = os.cpu_count()  # Not really a constant, but unlikely to change


class TrainingTransforms:
    def __init__(self, resize_to_dim: int):
        self.transforms = alb.Compose([
            alb.Resize(resize_to_dim, resize_to_dim),
            alb.RandomBrightnessContrast(),
            alb.RandomFog(),
            alb.RandomRain(),
            alb.Normalize(),  # default values are same as given in article
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


class ValidationTransforms:
    def __init__(self, resize_to_dim: int):
        self.transforms = alb.Compose([
            alb.Resize(resize_to_dim, resize_to_dim),
            alb.Normalize(),
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']


def get_datasets():
    dataset_train_transformed = datasets.ImageFolder(
        IMAGE_DIR,
        transform=(TrainingTransforms(IMAGE_DIM))
    )

    dataset_validation_transformed = datasets.ImageFolder(
        IMAGE_DIR,
        transform=(ValidationTransforms(IMAGE_DIM))
    )

    dataset_size = len(dataset_train_transformed)

    validation_size = int(VALID_SPLIT*dataset_size)

    rand_indicies = torch.randperm(len(dataset_train_transformed)).tolist()  # Random indicies for training and validation set

    # Think these slices should work?
    dataset_train = Subset(dataset_train_transformed, rand_indicies[validation_size:])
    dataset_vaild = Subset(dataset_validation_transformed, rand_indicies[:validation_size])

    return dataset_train, dataset_vaild, dataset_train_transformed.classes


def get_data_loaders(dataset_train, dataset_valid):
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    return train_loader, valid_loader

