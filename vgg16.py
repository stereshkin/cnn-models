import torch
from torch import nn

architecture = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M"
]


class VGG16(nn.Module):
    """Input shape: 224x224x3"""
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG16, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d((224, 224))
        self.conv_layers = self.__create_conv_layers(self.architecture)
        
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
        return x
        
        
    def __create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if isinstance(x, int):
                out_channels = x
                
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU())
                in_channels = x
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
 
