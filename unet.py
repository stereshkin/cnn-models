import torch
from torch import nn
from torchvision.transforms import CenterCrop


def _conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                         nn.ReLU(inplace=True)
                        )




class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Unet, self).__init__()
        self.conv_down1 = _conv_block(in_channels=in_channels, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_down2 = _conv_block(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_down3 = _conv_block(in_channels=128, out_channels=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_down4 = _conv_block(in_channels=256, out_channels=512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = _conv_block(in_channels=512, out_channels=1024)

        self.t1 = CenterCrop(size=56)
        self.t2 = CenterCrop(size=104)
        self.t3 = CenterCrop(size=200)
        self.t4 = CenterCrop(size=392)

        self.upsample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv_up1 = _conv_block(in_channels=1024, out_channels=512)
        self.upsample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_up2 = _conv_block(in_channels=512, out_channels=256)
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up3 = _conv_block(in_channels=256, out_channels=128)
        self.upsample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv_up4 = _conv_block(in_channels=128, out_channels=64)

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        conv1_output = self.conv_down1(x)
        conv2_output = self.conv_down2(self.maxpool1(conv1_output))
        conv3_output = self.conv_down3(self.maxpool2(conv2_output))
        conv4_output = self.conv_down4(self.maxpool3(conv3_output))

        conv5_output = self.conv5(self.maxpool4(conv4_output))

        upsample1_output = self.upsample1(conv5_output)
        conv4_output_cropped = self.t1(conv4_output)
        upsample1_output = torch.cat((conv4_output_cropped, upsample1_output), dim=1)

        conv1_up_output = self.conv_up1(upsample1_output)
        upsample2_output = self.upsample2(conv1_up_output)
        conv3_output_cropped = self.t2(conv3_output)
        upsample2_output = torch.cat((conv3_output_cropped, upsample2_output), dim=1)

        conv2_up_output = self.conv_up2(upsample2_output)
        upsample3_output = self.upsample3(conv2_up_output)
        conv2_output_cropped = self.t3(conv2_output)
        upsample3_output = torch.cat((conv2_output_cropped, upsample3_output), dim=1)

        conv3_up_output = self.conv_up3(upsample3_output)
        upsample4_output = self.upsample4(conv3_up_output)
        conv1_output_cropped = self.t4(conv1_output)
        upsample4_output = torch.cat((conv1_output_cropped, upsample4_output), dim=1)

        conv4_up_output = self.conv_up4(upsample4_output)
        output = self.output_conv(conv4_up_output)

        return output
