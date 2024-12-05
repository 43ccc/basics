import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.model(x)

class UpScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding),
            nn.Upsample(scale_factor=2, mode='nearest') # bilinear align_corners=False
        )
    
    def forward(self, x):
        return self.model(x)

# Skip-Layer-Excitation Module of FastGAN
# Modified for better memory management. 
# TODO: NEED TO TEST ACTUAL MEMORY IMPACT
class SLEModule(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(4,4)),
            nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=channels_out, out_channels=channels_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def reduce_input_tensor(self, x):
        self.reduced_input = self.model(x)

    def forward(self, x):
        return x * self.reduced_input

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()

        self.model = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
            ConvBlock(in_channels, in_channels, kernel_size=kernel_size, padding=padding),
        )
    
    def forward(self, x):
        return x + self.model(x)

# 160x90 -> 320x180 -> 640x360 -> 1280x720
# Loss: Epoch: 10 | Loss: 0.04315948776072926
# 7.6GB
# Test 
# lr schedule
# nn upsample
# 
# Epoch: 10 | Loss: 0.06468006215161748  
class ImageUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            #UpScaleBlock(3, out_channels=3),
            #UpScaleBlock(3, out_channels=3),
            ConvBlock(3, 32),
            ResNetBlock(32),
            ResNetBlock(32),
            ResNetBlock(32),
            ResNetBlock(32),
            UpScaleBlock(32, out_channels=8),
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def calc_loss(self, batch):
        loss_function = nn.L1Loss()
        downscaled = F.interpolate(batch, size=(360, 640), mode='bilinear', align_corners=False)
        loss = loss_function(batch, self(downscaled))

        return loss

