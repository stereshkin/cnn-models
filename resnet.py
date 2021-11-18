import torch.nn as nn


class res_block_2layers(nn.Module):
    expansion = 1

    def __init__(self, in_channels, intermediate_channels, downsample=None, stride=1):
        super(res_block_2layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_mapping = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            identity_mapping = self.downsample(x)

        output += identity_mapping
        output = self.relu(output)
        return output    
        
        
class res_block_3layers(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, intermediate_channels, downsample=None, stride=1):
        super(res_block_3layers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity_mapping = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)
        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.downsample is not None:
            identity_mapping = self.downsample(x)
            
        output += identity_mapping
        output = self.relu(output)
        return output
        
        
class Resnet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(Resnet, self).__init__()
        self.in_channels=64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, block, num_blocks, intermediate_channels, stride=1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != intermediate_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * block.expansion, kernel_size=1, stride = stride,
                          bias = False), 
                nn.BatchNorm2d(intermediate_channels * block.expansion))
        
        layers.append(block(self.in_channels, intermediate_channels, downsample, stride))
        self.in_channels = intermediate_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, intermediate_channels))
        
        return nn.Sequential(*layers)
        
        
def resnet18(in_channels=3, num_classes=1000):
    return Resnet(block=res_block_2layers, layers=[2, 2, 2, 2], image_channels=in_channels, num_classes=num_classes)

def resnet34(in_channels=3, num_classes=1000):
    return Resnet(block=res_block_2layers, layers=[3, 4, 6, 3], image_channels=in_channels, num_classes=num_classes)

def resnet50(in_channels=3, num_classes=1000):
    return Resnet(block=res_block_3layers, layers=[3, 4, 6, 3], image_channels=in_channels, num_classes=num_classes)

def resnet101(in_channels=3, num_classes=1000):
    return Resnet(block=res_block_3layers, layers=[3, 4, 23, 3], image_channels=in_channels, num_classes=num_classes)

def resnet152(in_channels=3, num_classes=1000):
    return Resnet(block=res_block_3layers, layers=[3, 8, 36, 3], image_channels=in_channels, num_classes=num_classes)        
