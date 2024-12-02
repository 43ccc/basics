import torch
import random
from torchvision import transforms, datasets
from networks import Classifier, Generator
from train import train

def main():
    batch_size = 32
    random_seed = 0

    # Fix random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Create data loaders
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    generator = Generator()
    discriminator = Classifier()

    train(train_loader, generator, discriminator, max_epochs=20, save_every_n_epochs=5)

if __name__ ==  '__main__':
    torch.manual_seed(0)
    main()
