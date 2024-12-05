import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

class ImageDataset(Dataset):
    def __init__(self, path_to_images, transform=transforms.ToTensor()):
        self.transform = transform
        self.image_paths = [f"{path_to_images}/{file}" for file in os.listdir(path_to_images)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
