import torch
import random
from train import train
from model import Classifier
from image_transform import get_transform
from data_functions import get_dataloaders
from hyperparameters import hyperparameters

# Device and seed
device = 'cuda' if torch.cuda.is_available else 'cpu'
RANDOM_SEED = 0

def main():

    # Fix random seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Transform
    transform=get_transform()
    
    # Get dataloader
    train_loader, test_loader = get_dataloaders(transform, batch_size=hyperparameters['batch_size'])

    # Init model
    model = Classifier()
    model.to(device)

    # Train the model
    train(num_epochs=hyperparameters['epochs'], train_dataloader=train_loader, test_dataloader=test_loader, model=model, lr=hyperparameters['lr'], device=device)  

if __name__ ==  '__main__':
    torch.manual_seed(0)
    main()
