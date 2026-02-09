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
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride_dw, padding=1, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride_pw, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_pw = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu_dw(x)

        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu_pw(x)

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
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64, 1, 1),
            DepthwiseSeparableConv(64, 128, 2, 1),
            DepthwiseSeparableConv(128, 128, 1, 1),
            DepthwiseSeparableConv(128, 256, 2, 1),
            DepthwiseSeparableConv(256, 256, 1, 1),
            DepthwiseSeparableConv(256, 512, 2, 1),
            DepthwiseSeparableConv(512, 512, 1, 1),
            DepthwiseSeparableConv(512, 512, 1, 1),
            DepthwiseSeparableConv(512, 512, 1, 1),
            DepthwiseSeparableConv(512, 512, 1, 1),
            DepthwiseSeparableConv(512, 512, 1, 1),
            DepthwiseSeparableConv(512, 1024, 2, 1),
            DepthwiseSeparableConv(1024, 1024, 1, 1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )


    def forward(self, x):
        out = self.model(x)
        return out