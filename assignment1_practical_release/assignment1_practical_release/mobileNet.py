import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride_dw, stride_pw):
        super().__init__()

        """
        Build the depthwise separable convolution layer
        For the depthwise convolution (use padding=1 and bias=False for the convolution)
        For the pointwise convolution (use padding=0 and bias=False fot the convolution)

        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            stride_dw: stride for depthwise convolution
            stride_pw: stride for pointwise convolution
        """
        
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride_dw,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride_pw,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


    

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        Build the MobileNet architecture
        For the first standard convolutional layer (use padding=1 and bias=False for the convolution)
        For the AvgPool layer, use nn.AdaptiveAvgPool2d.

        Inputs:
            num_classes: number of classes for classification
        """

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.dw_sep_conv0 = DepthwiseSeparableConv(32, 64, 1, 1)
        self.dw_sep_conv1 = DepthwiseSeparableConv(64, 128, 2, 1)
        self.dw_sep_conv2 = DepthwiseSeparableConv(128, 128, 1, 1)
        self.dw_sep_conv3 = DepthwiseSeparableConv(128, 256, 2, 1)
        self.dw_sep_conv4 = DepthwiseSeparableConv(256, 256, 1, 1)
        self.dw_sep_conv5 = DepthwiseSeparableConv(256, 512, 2, 1)

        self.dw_sep_conv61 = DepthwiseSeparableConv(512, 512, 1, 1)
        self.dw_sep_conv62 = DepthwiseSeparableConv(512, 512, 1, 1)
        self.dw_sep_conv63 = DepthwiseSeparableConv(512, 512, 1, 1)
        self.dw_sep_conv64 = DepthwiseSeparableConv(512, 512, 1, 1)
        self.dw_sep_conv65 = DepthwiseSeparableConv(512, 512, 1, 1)

        self.dw_sep_conv7 = DepthwiseSeparableConv(512, 1024, 2, 1)
        self.dw_sep_conv8 = DepthwiseSeparableConv(1024, 1024, 1, 1)

        self.avgpool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)

        x = self.dw_sep_conv0(x)
        x = self.dw_sep_conv1(x)
        x = self.dw_sep_conv2(x)
        x = self.dw_sep_conv3(x)
        x = self.dw_sep_conv4(x)
        x = self.dw_sep_conv5(x)

        x = self.dw_sep_conv61(x)
        x = self.dw_sep_conv62(x)
        x = self.dw_sep_conv63(x)
        x = self.dw_sep_conv64(x)
        x = self.dw_sep_conv65(x)

        x = self.dw_sep_conv7(x)
        x = self.dw_sep_conv8(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
