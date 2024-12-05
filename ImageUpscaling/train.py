import torch
import os
from torchvision.utils import save_image
import torch.nn.functional as F

@torch.no_grad()
def visualize_results(model, num_images, dataloader, device, path='generated_images'):
    os.makedirs(path, exist_ok=True)  
    model.eval()

    left_to_generate = num_images

    for batch in dataloader:
        batch = batch.to(device)
        downscaled = F.interpolate(batch, size=(360, 640), mode='bilinear', align_corners=False)
        images = model(downscaled)

        num_to_save = min(left_to_generate, len(images))

        # Save the images to disk
        for i in range(num_to_save):
            original_image_path = os.path.join(path, f'image_{num_images - left_to_generate + i + 1}_original.png')
            image_path = os.path.join(path, f'image_{num_images - left_to_generate + i + 1}.png')
            save_image(images[i], image_path)
            save_image(downscaled[i], original_image_path)

        left_to_generate -= num_to_save
        if left_to_generate <= 0:
            break

def train(model, dataloader, epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    scaler = torch.amp.grad_scaler.GradScaler(device)

    model.train()

    for e in range(epochs):
        epoch_loss = 0

        for batch in dataloader:
            batch = batch.to(device)
            #batch = F.interpolate(batch, size=(180, 320), mode='bilinear', align_corners=False)

            with torch.amp.autocast(device):
                loss = model.calc_loss(batch)

            epoch_loss += loss.item()

            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        epoch_loss /= len(dataloader)
        print(f'Epoch: {e+1} | Loss: {epoch_loss}')

    visualize_results(model, num_images=10, dataloader=dataloader, device=device)

