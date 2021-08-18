import torch
from torch import nn


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, x):
        return self.leaky_relu(self.batchnorm(self.conv(x)))


class Yolo(nn.Module):
    def __init__(self, in_channels=3, grid_size=7, num_boxes=2, num_classes=20):
        super(Yolo, self).__init__()
        self._in_channels = in_channels
        self._grid_size = grid_size
        self._num_boxes = num_boxes
        self._num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((448, 448))
        self.architecture = architecture_config
        self.conv_part = self._create_conv_layers(
            architecture=self.architecture,
            in_channels=self._in_channels
            )
        self.fcs = self._create_fc_layers(
            grid_size=self._grid_size,
            num_boxes=self._num_boxes,
            num_classes=self._num_classes
            )
        
    def forward(self, x):
        S = self._grid_size
        B = self._num_boxes
        C = self._num_classes 
        x = self.avgpool(x)
        x = self.conv_part(x)
        x = x.view(x.shape[0], -1)
        x = self.fcs(x)
        x = x.view(-1, S, S, C + B * 5)
        return x

    @staticmethod    
    def _create_fc_layers(grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes
        return nn.Sequential(nn.Linear(1024 * S * S, 4096), 
                             nn.Dropout(p=0.5),
                             nn.LeakyReLU(negative_slope=0.1),
                             nn.Linear(4096, S * S * (C + B * 5)),
                             nn.ReLU())

    @staticmethod           
    def _create_conv_layers(architecture, in_channels):
        layers = list()
        for x in architecture:
            if isinstance(x, tuple):
                kernel_size = x[0]
                out_channels = x[1]
                stride = x[2]
                padding = x[3]
                layers.append(block(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding))
                in_channels = x[1]
                
            elif isinstance(x, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                
            elif isinstance(x, list):
                layer1 = x[0]
                layer2 = x[1]
                times = x[2]
                
                for _ in range(times):
                    layers.append(block(in_channels=in_channels,
                                        out_channels=layer1[1],
                                        kernel_size=layer1[0],
                                        stride=layer1[2],
                                        padding=layer1[3]))
                    layers.append(block(in_channels=layer1[1],
                                        out_channels=layer2[1],
                                        kernel_size=layer2[0],
                                        stride=layer2[2],
                                        padding=layer2[3]))
                    in_channels = layer2[1]
          
        return nn.Sequential(*layers)
        
