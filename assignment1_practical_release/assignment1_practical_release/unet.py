import torch
from torch import nn


def double_conv_block(in_channels, out_channels):
    """
    This double conv block are the blocks used in the encoder part of UNet.
    It uses a padding of 1 to preserve spatial dimensions.
    
    :param in_channels: Description
    :param out_channels: Description
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    """
    Decoder block of UNet. It consists of an upconvolution layer followed by a double conv block.
    Use the double_conv_block defined above.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv_block(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        in_channels = input_shape

        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)

        self.encoder_block5 = double_conv_block(512, 1024)

        # Decoder blocks
        self.decoder_block1 = DecoderBlock(1024, 512)
        self.decoder_block2 = DecoderBlock(512, 256)
        self.decoder_block3 = DecoderBlock(256, 128)
        self.decoder_block4 = DecoderBlock(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder with skip connections
        s1 = self.encoder_block1(x)
        p1 = self.pool(s1)
        
        s2 = self.encoder_block2(p1)
        p2 = self.pool(s2)
        
        s3 = self.encoder_block3(p2)
        p3 = self.pool(s3)
        
        s4 = self.encoder_block4(p3)
        p4 = self.pool(s4)

        # Bottleneck
        b = self.encoder_block5(p4)

        # Decoder
        d1 = self.decoder_block1(b, s4)
        d2 = self.decoder_block2(d1, s3)
        d3 = self.decoder_block3(d2, s2)
        d4 = self.decoder_block4(d3, s1)

        # Output
        outputs = self.outconv(d4)
        return outputs