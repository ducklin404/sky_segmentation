import torch
import torch.nn as nn
from decoder import UpBlock
from encoder import MobileNetV3Encoder

class MobileUNetLite(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.encoder = MobileNetV3Encoder(pretrained=True)

        # start up sample to get the mask
        # here we use 4 only to keep the model small and fast enough
        self.up4 = UpBlock(576, 96, 256)
        self.up3 = UpBlock(256, 40, 128)
        self.up2 = UpBlock(128, 24, 64)
        self.up1 = UpBlock(64, 16, 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        d4 = self.up4(x5, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        return self.final(d1)
