import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from data_functions import get_dataloaders
from train import train
from data_transforms import get_test_transform, get_train_transform

def main():
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    train_loader, test_loader = get_dataloaders(train_transform=get_train_transform(), test_transform=get_test_transform(), batch_size=32)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_loader.dataset.classes))
    model.to(device)
    train(num_epochs=10, train_dataloader=train_loader, test_dataloader=test_loader, model=model, lr=0.001, device=device)

if __name__ =='__main__':
    main()