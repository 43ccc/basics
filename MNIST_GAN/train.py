import torch
import torch.optim as optim
from util import save_model, save_image_to_disk


def train(dataloader, generator, discriminator, max_epochs, save_every_n_epochs=None, gen_lr=0.0002, dis_lr=0.0002):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()
    
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=dis_lr)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr)

    for epoch in range(max_epochs):
        dis_epoch_loss = 0
        gen_epoch_loss = 0

        for images, _ in dataloader:
            images = images.to(device)

            # Train discriminator
            dis_loss = discriminator.calc_loss(images, generator)
            dis_epoch_loss += dis_loss.item()
            dis_loss.backward()
            dis_optimizer.step()
            dis_optimizer.zero_grad()

            # Train generator
            gen_loss = generator.calc_loss(images.size(0), discriminator)
            gen_epoch_loss += gen_loss.item()
            gen_loss.backward()
            gen_optimizer.step()
            gen_optimizer.zero_grad()
        
        dis_epoch_loss /= len(dataloader)
        gen_epoch_loss /= len(dataloader)

        print(f'Epoch: {epoch+1} | Discriminator loss: {dis_epoch_loss} | Generator loss: {gen_epoch_loss}')

        # Save model every n epochs
        if epoch % save_every_n_epochs == 0 or epoch == max_epochs-1:
            save_model(generator, path='./trained_models', file_name=f'generator_{epoch+1}')
            save_model(discriminator, path='./trained_models', file_name=f'discriminator_{epoch+1}')

            save_image_to_disk(generator, path=f'./generated_images/', file_name=f'{epoch+1}.png')

        