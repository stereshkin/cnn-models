import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli


class stochastic_res_block_2layers(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, intermediate_channels, prob, train, downsample=None, stride=1):
        super(stochastic_res_block_2layers, self).__init__()
        self.is_train = train
        self.prob = prob
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
        
        if self.is_train:
            m = Bernoulli(torch.tensor(self.prob))
            result = m.sample().item()
            if result == 0:
                output *= torch.as_tensor(result)   
        else:
            output *= torch.as_tensor(self.prob)        
            
        output += identity_mapping
        output = self.relu(output)
        return output 

    
class stochastic_res_block_3layers(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, intermediate_channels, prob, train, downsample=None, stride=1):
        super(stochastic_res_block_3layers, self).__init__()
        self.is_train = train
        self.prob = prob
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
        
        if self.is_train:
            m = Bernoulli(torch.tensor(self.prob))
            result = m.sample().item()
            if result == 0:
                output *= torch.as_tensor(result)
        else:
            output *= torch.as_tensor(self.prob)         
            
        output += identity_mapping
        output = self.relu(output)
        return output


class StochasticResnet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, prob=0.5):
        super(StochasticResnet, self).__init__()
        self.prob = prob
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64, train=self.training, prob=self.prob)
        self.layer2 = self._make_layer(block, layers[1], 128, train=self.training, prob=self.prob, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, train=self.training, prob=self.prob, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, train=self.training, prob=self.prob, stride=2)
        
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
        
    def _make_layer(self, block, num_blocks, intermediate_channels, train, prob, stride=1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != intermediate_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * block.expansion, kernel_size=1, stride = stride,
                          bias = False), 
                nn.BatchNorm2d(intermediate_channels * block.expansion))
        
        layers.append(block(self.in_channels, intermediate_channels, prob, train, downsample, stride))
        self.in_channels = intermediate_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, intermediate_channels, prob, train))
        
        return nn.Sequential(*layers)

    
def stochastic_resnet18(in_channels=3, num_classes=1000, prob=0.5):
    return StochasticResnet(
        block=stochastic_res_block_2layers,
        layers=[2, 2, 2, 2],
        image_channels=in_channels,
        num_classes=num_classes,
        prob=prob
    )

def stochastic_resnet34(in_channels=3, num_classes=1000, prob=0.5):
    return StochasticResnet(
        block=stochastic_res_block_2layers,
        layers=[3, 4, 6, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        prob=prob
    )

def stochastic_resnet50(in_channels=3, num_classes=1000, prob=0.5):
    return StochasticResnet(
        block=stochastic_res_block_3layers,
        layers=[3, 4, 6, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        prob=prob
    )

def stochastic_resnet101(in_channels=3, num_classes=1000, prob=0.5):
    return StochasticResnet(
        block=stochastic_res_block_3layers,
        layers=[3, 4, 23, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        prob=prob
    )

def stochastic_resnet152(in_channels=3, num_classes=1000, prob=0.5):
    return StochasticResnet(
        block=stochastic_res_block_3layers,
        layers=[3, 8, 36, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        prob=prob
    )    
