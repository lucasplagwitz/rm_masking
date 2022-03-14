""" Full assembly of the parts to form the complete network """
# Architecture follows public repository:
# https://github.com/milesial/Pytorch-UNet
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        f = 1
        self.inc = DoubleConv(n_channels, 32*f) # 64
        self.down0 = Down(32*f, 64*f)
        self.down1 = Down(64*f, 128*f)
        self.down2 = Down(128*f, 256*f)
        self.down3 = Down(256*f, 512*f)
        factor = 2 if bilinear else 1
        self.down4 = Down(512*f, 1024*f // factor)
        self.up1 = Up(1024*f, 512 *f// factor, bilinear)
        self.up2 = Up(512*f, 256*f // factor, bilinear)
        self.up3 = Up(256*f, 128*f // factor, bilinear)
        self.up4 = Up(128*f, 64*f // factor, bilinear)
        self.up5 = Up(64*f, 32*f, bilinear)
        self.outc = OutConv(32*f, n_classes)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down0(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x0)
        logits = self.outc(x)
        return logits
