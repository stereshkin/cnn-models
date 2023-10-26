import torch
import torch.nn as nn


class DCGAN_Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        channels = [[1024, 512], [512, 256], [256, 128], [128, 3]]
        self.project = nn.Linear(100, 4 * 4 * 1024)
        self.conv_layers = self._create_conv_layers(channels) 
    
    @staticmethod
    def _create_conv_layers(channels):
        layers = []
        for i in range(3):
            layers.extend([
                nn.ConvTranspose2d(in_channels=channels[i][0], out_channels=channels[i][1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(channels[i][1]),
                nn.ReLU()
            ])
        layers.append(nn.ConvTranspose2d(in_channels=channels[-1][0], out_channels=channels[-1][1], kernel_size=4, stride=2, padding=1))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
            
    
    def forward(self, z):
        z = self.project(z)
        z = z.view(z.size(0), 1024, 4, 4)
        z = self.conv_layers(z)
        return z
