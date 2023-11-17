import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
            
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        x_block = self.block(x)
        return x + x_block


class Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )
        
        blocks = [ResBlock(channels=256) for _ in range(9)]
        self.blocks = nn.Sequential(*blocks)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        x = self.upsample(x)
        return x
      
