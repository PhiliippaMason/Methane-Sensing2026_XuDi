import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Base MLP module with ReLU activation
    Parameters:
    -----------
    in_channels: Int
        Number of input channels
    out_channels: Int
        Number of output channels
    h_channels: Int
        Number of hidden channels
    h_layers: Int
        Number of hidden layers
    """

    def __init__(self, in_channels, out_channels, h_channels=64, h_layers=4):

        super().__init__()

        def hidden_block(h_channels):
            h = nn.Sequential(nn.Linear(h_channels, h_channels), nn.ReLU())
            return h

        # Model

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            nn.ReLU(),
            *[hidden_block(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        div_factor=8,
        prob_output=True,
        class_output=False,
    ):
        super(Unet, self).__init__()

        self.n_channels = in_channels
        self.bilinear = True
        self.sigmoid = nn.Sigmoid()
        self.prob_output = prob_output
        self.class_output = class_output

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2), double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    self.up = nn.ConvTranpose2d(
                        in_channels // 2, in_channels // 2, kernel_size=2, stride=2
                    )

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )
                x = torch.cat([x2, x1], dim=1)  ## why 1?
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64 // div_factor)
        self.down1 = down(64 // div_factor, 128 // div_factor)
        self.down2 = down(128 // div_factor, 256 // div_factor)
        self.down3 = down(256 // div_factor, 512 // div_factor)
        self.down4 = down(512 // div_factor, 512 // div_factor)
        self.up1 = up(1024 // div_factor, 256 // div_factor)
        self.up2 = up(512 // div_factor, 128 // div_factor)
        self.up3 = up(256 // div_factor, 64 // div_factor)
        self.up4 = up(128 // div_factor, 128 // div_factor)
        self.out = nn.Conv2d(128 // div_factor, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # return self.out(x).permute(0,2,3,1)

        if self.prob_output:
            x = self.out(x)
            return self.sigmoid(x).permute(0, 2, 3, 1)
        else:
            return self.out(x).permute(0, 2, 3, 1)


# quantitive
class DeepUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DeepUNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.encoder1 = CBR(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = CBR(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.encoder5 = CBR(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

        self.encoder6 = CBR(1024, 2048)
        self.pool6 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = CBR(2048, 4096)

        # Decoder
        self.upconv6 = nn.ConvTranspose2d(
            4096, 2048, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder6 = CBR(4096, 2048)

        self.upconv5 = nn.ConvTranspose2d(
            2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder5 = CBR(2048, 1024)

        self.upconv4 = nn.ConvTranspose2d(
            1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder4 = CBR(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder3 = CBR(512, 256)

        self.upconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.decoder1 = CBR(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool6(enc6))

        # Decoding path
        dec6 = self.upconv6(bottleneck)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)

        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1).permute(0, 2, 3, 1)
