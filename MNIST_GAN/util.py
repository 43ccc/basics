import torch
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Save model state dicts to path
def save_model(model, path, file_name):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    torch.save(model.state_dict(), full_path)

def save_image_to_disk(model, path, file_name, num_images=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, file_name)
    images = model(torch.randn(num_images, model.latent_size, 1, 1, device=device))
    save_image(images, full_path)