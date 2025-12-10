import torch.nn as nn
from torchvision import models

# pretrained MobileNetV3-small as the encoder
class MobileNetV3Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()


        # get all features
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        ).features

    
        
        # extract data before down sample for later skip connection
        # stop after the last bottleneck block, before the inflate 1x1

        self.stage1 = base[:1]    # 16 channels, 1/2
        self.stage2 = base[1:4]   # 24 channels, 1/4
        self.stage3 = base[4:9]   # 40 channels, 1/8
        self.stage4 = base[9:12]  # 96 channels, 1/16
        

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        return x1, x2, x3, x4


if __name__ == '__main__':
    base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    for name, layer in base.features.named_children():
        print(name, layer)