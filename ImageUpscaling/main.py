from dataset import ImageDataset
from train import train
from model import ImageUpscaler
import torch 
import random

RANDOM_SEED = 0

def main():
    # Fix random seed
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    dataset = ImageDataset(path_to_images='./data/test')
    dloader = torch.utils.data.DataLoader(dataset, 16, shuffle=True, pin_memory=True)
    model = ImageUpscaler()
    model.to('cuda')

    train(model, dloader, epochs=20)

if __name__ == '__main__':
    main()
