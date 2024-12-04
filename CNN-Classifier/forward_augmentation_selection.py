import torchvision.transforms.v2 as transforms
from train import train
from data_functions import get_dataloaders
from model import Classifier
from hyperparameters import hyperparameters

def get_model(device='cuda'):
    model = Classifier()
    model.to(device)

    return model

def forward_selct_augmenations(base_augmentations, list_of_augmentations, device):
    current_augmentations = base_augmentations

    model = get_model(device)
    train_loader, test_loader = get_dataloaders(current_augmentations, hyperparameters['batch_size'])
    best_acc = train(hyperparameters['epochs'], train_loader, test_loader, model, hyperparameters['lr'], device)

    for i in range(len(list_of_augmentations)):
        model = get_model(device)
        current_augmentations = current_augmentations + list_of_augmentations[i:i+1]
        train_loader, test_loader = get_dataloaders(current_augmentations, hyperparameters['batch_size'])
        curr_acc = train(hyperparameters['epochs'], train_loader, test_loader, model, hyperparameters['lr'], device)

        if not curr_acc > best_acc:
            current_augmentations = current_augmentations[:-1]
    
    return current_augmentations


if __name__ =='__main__':
    possible_transforms = [
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(32,32), scale=(0.9, 1)),
        transforms.GaussianNoise(mean=0, sigma=0.001, clip=True),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random', inplace=True)
    ]

    base_transforms = [
        transforms.ToTensor()
    ]

    selected_augmentations = forward_selct_augmenations(base_transforms, possible_transforms, device='cuda')
    print(selected_augmentations)
    
    