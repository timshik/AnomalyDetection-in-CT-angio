import torch
import torch.nn as nn
import torch.nn.functional as F

layer1_filters = 8


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear):
        super().__init__()
        factor = 2 if bilinear else 1
        self.n_channels = n_channels
        self.inc = DoubleConv(n_channels, layer1_filters)
        self.down1 = Down(layer1_filters, layer1_filters*2)
        self.down2 = Down(layer1_filters*2, layer1_filters*4)
        self.down3 = Down(layer1_filters*4, layer1_filters*8)
        self.down4 = Down(layer1_filters*8, layer1_filters*16 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


class Decoder(nn.Module):
    def __init__(self, n_classes, bilinear):
        super().__init__()
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        self.up1 = Up(layer1_filters*16, layer1_filters*8 // factor, bilinear)
        self.up2 = Up(layer1_filters*8, layer1_filters*4 // factor, bilinear)
        self.up3 = Up(layer1_filters*4, layer1_filters*2 // factor, bilinear)
        self.up4 = Up(layer1_filters*2, layer1_filters, bilinear)
        self.outc = OutConv(layer1_filters, n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.encoder = Encoder(n_channels, bilinear)
        self.decoder = Decoder(n_classes, bilinear)

    def forward(self, x):
        return self.decoder(*self.encoder(x))

    def save(self, path_enc, path_dec):
        self.encoder.save(path_enc)
        self.decoder.save(path_dec)

    def load(self, path_enc, path_dec):
        self.encoder.load(path_enc)
        self.decoder.load(path_dec)
