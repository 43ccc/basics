import torch
import torch.nn as nn
import torch.nn.functional as F

# Classifier
class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.out = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7,7), stride=1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.out(self.fc(x))

        return x

    def calc_loss(self, real_images, generator):
        device = 'cuda' if torch.cuda.is_available else 'cpu'

        with torch.no_grad():
            fake_images = generator(torch.randn(real_images.size(0), generator.latent_size, 1, 1, device=device))

        pred_real = self(real_images)
        pred_fake = self(fake_images)

        label_real = torch.ones_like(pred_real)
        label_fake = torch.zeros_like(pred_fake)

        loss_real = F.binary_cross_entropy(pred_real, label_real)
        loss_fake = F.binary_cross_entropy(pred_fake, label_fake)

        return loss_real + loss_fake

# Generator
class Generator(nn.Module):

    def __init__(self, latent_size=256):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.out = nn.Sigmoid()

        self.conv1 = nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size // 2, kernel_size=(7,7), stride=1)
        self.conv2 = UpSampleLayer(in_channels=latent_size // 2, out_channels=latent_size // 4)
        self.conv3 = UpSampleLayer(in_channels=latent_size // 4, out_channels=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.out(self.conv3(x))

        return x
    
    def calc_loss(self, batch_size, discriminator):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fake_images = self(torch.randn(batch_size, self.latent_size, 1, 1, device=device))
        
        pred = discriminator(fake_images)
        label_real = torch.ones_like(pred)

        loss = F.binary_cross_entropy(pred, label_real)

        return loss

# Submodules
class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSampleLayer, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.up(x)
        return x