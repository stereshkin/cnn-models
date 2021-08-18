import torch
from torch import nn

class Alexnet(nn.Module):
    def __init__(self, in_channels=3, out_features=1000):
        super(Alexnet, self).__init__()    
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fcs = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=4096, out_features=out_features)
        )
        
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
        return x

