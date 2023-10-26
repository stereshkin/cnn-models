import torch
import torch.nn as nn


class DCGAN_Discriminator(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv = self._create_conv_layers(in_channels, 100)
        self.out = nn.Sequential(nn.Linear(4900, 1), nn.Sigmoid())
            
    @staticmethod
    def _create_conv_layers(in_channels, mid_channels):
        layers = [nn.Conv2d(in_channels, 100, kernel_size=3, stride=2)]
            
        for _ in range(2):
            layers.extend([
                nn.Conv2d(100, 100, kernel_size=3, stride=2),
                nn.BatchNorm2d(100),
                nn.LeakyReLU(0.2),
            ])
        
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.out(x)
