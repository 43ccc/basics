import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

def get_dataloaders(augmentations, batch_size, pin_memory=torch.cuda.is_available()):
    transform = transforms.Compose(augmentations)

    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=pin_memory)

    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, pin_memory=pin_memory)

    return train_loader, test_loader