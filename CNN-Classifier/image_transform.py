import torchvision.transforms.v2 as transforms
import torch

def get_transform():
    transform = [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(32,32), scale=(0.9, 1)),
        transforms.GaussianNoise(mean=0, sigma=0.001, clip=True),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random', inplace=True)
        ]

    return transform