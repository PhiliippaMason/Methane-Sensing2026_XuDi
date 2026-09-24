#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
from network.VSSBlock import SS2D, Mlp
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        guiding_map0 = F.interpolate(
            guiding_map0, x.size()[2:], mode="bilinear", align_corners=True
        )

        guiding_map = F.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class N_BPMSNet_Lite_v1(nn.Module):
    def __init__(self):
        super(N_BPMSNet_Lite_v1, self).__init__()

        # Smaller backbone: Use MobileNetV2 as a lightweight feature extractor
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.load_state_dict(torch.load("./network/mobilenet_v2-b0353104.pth"))
        self.features = copy.deepcopy(mobilenet.features[:7])
        self.features_1 = copy.deepcopy(mobilenet.features[:7])
        old_conv = self.features[0][0]
        new_conv = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
        self.features[0][0] = new_conv
        self.features_1[0][0] = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.features_1[0][0].weight = old_conv.weight

        # Downsampled feature layers
        self.down1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.CDdown1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.CDdown2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_reduce_1 = BasicConv2d(32 * 3, 32, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 3, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(128 * 3, 128, 3, 1, 1)

        # Decoder with reduced channels
        self.decoder_module3 = BasicConv2d(192, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(96, 32, 3, 1, 1)
        self.decoder_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B, C):
        # A: target date images; B: reference date images; C: NDMI images
        size = A.size()[2:]

        # Process input images through modified MobileNet backbone
        layer1_A = self.features(A)
        layer1_B = self.features(B)
        layer1_C = self.features_1(C)

        # Process downsampled layers
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)

        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)

        layer2_C = self.CDdown1(layer1_C)
        layer3_C = self.CDdown2(layer2_C)

        # Concatenate for comparison, followed by reduction
        combined_layer1 = torch.cat((layer1_A, layer1_B, layer1_C), dim=1)
        combined_layer2 = torch.cat((layer2_A, layer2_B, layer2_C), dim=1)
        combined_layer3 = torch.cat((layer3_A, layer3_B, layer3_C), dim=1)
        combined_layer1 = self.conv_reduce_1(combined_layer1)
        combined_layer2 = self.conv_reduce_2(combined_layer2)
        combined_layer3 = self.conv_reduce_3(combined_layer3)

        # Apply Change Guide Modules
        feature3 = self.decoder_module3(
            torch.cat([self.upsample2x(combined_layer3), combined_layer2], 1)
        )
        feature2 = self.decoder_module2(
            torch.cat([self.upsample2x(feature3), combined_layer1], 1)
        )
        # Final decoding layer
        final_map = self.decoder_final(feature2)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)

        return None, None, None, final_map


class N_BPMSNet_Lite_v2(nn.Module):
    def __init__(self):
        super(N_BPMSNet_Lite_v2, self).__init__()

        # Smaller backbone: Use MobileNetV2 as a lightweight feature extractor
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.load_state_dict(torch.load("./network/mobilenet_v2-b0353104.pth"))
        self.features = copy.deepcopy(mobilenet.features[:7])
        self.features_1 = copy.deepcopy(mobilenet.features[:7])
        old_conv = self.features[0][0]
        new_conv = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
        self.features[0][0] = new_conv
        self.features_1[0][0] = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.features_1[0][0].weight = old_conv.weight

        # Downsampled feature layers
        self.down1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.CDdown1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.CDdown2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_reduce_1 = BasicConv2d(32 * 4, 32, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 4, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(128 * 4, 128, 3, 1, 1)

        self.conv_reduce_4 = BasicConv2d(32 * 2, 32, 3, 1, 1)
        self.conv_reduce_5 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_6 = BasicConv2d(128 * 2, 128, 3, 1, 1)

        # Decoder with reduced channels
        self.decoder_module3 = BasicConv2d(192, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(96, 32, 3, 1, 1)
        self.decoder_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B, C):
        # A: target date images; B: reference date images; C: NDMI images
        size = A.size()[2:]

        # Process input images through modified MobileNet backbone
        layer1_A = self.features(A)
        layer1_B = self.features(B)
        layer1_C = self.features_1(C)

        # Process downsampled layers
        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)

        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)

        layer2_C = self.CDdown1(layer1_C)
        layer3_C = self.CDdown2(layer2_C)

        # Cross Channel Module
        B, C, H, W = layer1_A.size()
        layer1_D = torch.empty(B, C * 2, H, W)  # .cuda()
        layer1_D[:, ::2, :, :] = layer1_A
        layer1_D[:, 1::2, :, :] = layer1_B
        combined_layer1 = torch.cat((layer1_A, layer1_B, layer1_D), dim=1)

        B, C, H, W = layer2_A.size()
        layer2_D = torch.empty(B, C * 2, H, W)  # .cuda()
        layer2_D[:, ::2, :, :] = layer2_A
        layer2_D[:, 1::2, :, :] = layer2_B
        combined_layer2 = torch.cat((layer2_A, layer2_B, layer2_D), dim=1)

        B, C, H, W = layer3_A.size()
        layer3_D = torch.empty(B, C * 2, H, W)  # .cuda()
        layer3_D[:, ::2, :, :] = layer3_A
        layer3_D[:, 1::2, :, :] = layer3_B
        combined_layer3 = torch.cat((layer3_A, layer3_B, layer3_D), dim=1)
        combined_layer1 = self.conv_reduce_1(combined_layer1)
        combined_layer2 = self.conv_reduce_2(combined_layer2)
        combined_layer3 = self.conv_reduce_3(combined_layer3)

        # Concatenate for comparison, followed by reduction
        combined_layer4 = torch.cat((combined_layer1, layer1_C), dim=1)
        combined_layer5 = torch.cat((combined_layer2, layer2_C), dim=1)
        combined_layer6 = torch.cat((combined_layer3, layer3_C), dim=1)
        combined_layer4 = self.conv_reduce_4(combined_layer4)
        combined_layer5 = self.conv_reduce_5(combined_layer5)
        combined_layer6 = self.conv_reduce_6(combined_layer6)

        # Apply Change Guide Modules
        feature3 = self.decoder_module3(
            torch.cat([self.upsample2x(combined_layer6), combined_layer5], 1)
        )
        feature2 = self.decoder_module2(
            torch.cat([self.upsample2x(feature3), combined_layer4], 1)
        )
        # Final decoding layer
        final_map = self.decoder_final(feature2)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)

        return None, None, None, final_map


class N_BPMSNet_Lite_v3(nn.Module):
    def __init__(self):
        super(N_BPMSNet_Lite_v3, self).__init__()

        # Smaller backbone: Use MobileNetV2 as a lightweight feature extractor
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.load_state_dict(torch.load("./network/mobilenet_v2-b0353104.pth"))
        self.features = copy.deepcopy(mobilenet.features[:7])
        self.features_1 = copy.deepcopy(mobilenet.features[:7])
        old_conv = self.features[0][0]
        new_conv = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
        self.features[0][0] = new_conv
        self.features_1[0][0] = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.features_1[0][0].weight = old_conv.weight

        # Downsampled feature layers
        self.down1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.CDdown1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.CDdown2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_reduce_1 = BasicConv2d(32 * 2, 32, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(128 * 2, 128, 3, 1, 1)

        self.CDdecoder1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.CDdecoder2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.CDdecoder3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.decoder_module3 = BasicConv2d(192, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(96, 32, 3, 1, 1)
        self.decoder_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

        self.cgm_1 = ChangeGuideModule(32)
        self.cgm_2 = ChangeGuideModule(64)
        self.cgm_3 = ChangeGuideModule(128)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B, C):
        size = A.size()[2:]

        layer1_A = self.features(A)
        layer1_B = self.features(B)
        layer1_C = self.features_1(C)

        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)

        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)

        layer2_C = self.CDdown1(layer1_C)
        layer3_C = self.CDdown2(layer2_C)

        combined_layer1 = torch.cat((layer1_A, layer1_B), dim=1)
        combined_layer2 = torch.cat((layer2_A, layer2_B), dim=1)
        combined_layer3 = torch.cat((layer3_A, layer3_B), dim=1)
        combined_layer1 = self.conv_reduce_1(combined_layer1)
        combined_layer2 = self.conv_reduce_2(combined_layer2)
        combined_layer3 = self.conv_reduce_3(combined_layer3)

        change_map1 = self.CDdecoder1(layer1_C)
        change_map2 = self.CDdecoder2(layer2_C)
        change_map3 = self.CDdecoder3(layer3_C)

        layer3_out = self.cgm_3(combined_layer3, change_map3)
        feature3 = self.decoder_module3(
            torch.cat([self.upsample2x(layer3_out), combined_layer2], 1)
        )
        layer2_out = self.cgm_2(feature3, change_map2)
        feature2 = self.decoder_module2(
            torch.cat([self.upsample2x(layer2_out), combined_layer1], 1)
        )
        layer1_out = self.cgm_1(feature2, change_map1)

        final_map = self.decoder_final(layer1_out)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)
        change_maps3 = F.interpolate(
            change_map3, size, mode="bilinear", align_corners=True
        )
        change_maps2 = F.interpolate(
            change_map2, size, mode="bilinear", align_corners=True
        )
        change_maps1 = F.interpolate(
            change_map1, size, mode="bilinear", align_corners=True
        )

        return change_maps3, change_maps2, change_maps1, final_map


class N_BPMSNet(nn.Module):
    def __init__(self):
        super(N_BPMSNet, self).__init__()

        # Smaller backbone: Use MobileNetV2 as a lightweight feature extractor
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.load_state_dict(torch.load("./network/mobilenet_v2-b0353104.pth"))
        self.features = copy.deepcopy(mobilenet.features[:7])
        self.features_1 = copy.deepcopy(mobilenet.features[:7])
        old_conv = self.features[0][0]
        new_conv = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
        self.features[0][0] = new_conv
        self.features_1[0][0] = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.features_1[0][0].weight = old_conv.weight

        # Downsampled feature layers
        self.down1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.CDdown1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.CDdown2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_reduce_1 = BasicConv2d(32 * 4, 32, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 4, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(128 * 4, 128, 3, 1, 1)

        self.CDdecoder1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.CDdecoder2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.CDdecoder3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.decoder_module3 = BasicConv2d(192, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(96, 32, 3, 1, 1)
        self.decoder_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

        self.cgm_1 = ChangeGuideModule(32)
        self.cgm_2 = ChangeGuideModule(64)
        self.cgm_3 = ChangeGuideModule(128)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B, C):
        size = A.size()[2:]

        layer1_A = self.features(A)
        layer1_B = self.features(B)
        layer1_C = self.features_1(C)

        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)

        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)

        layer2_C = self.CDdown1(layer1_C)
        layer3_C = self.CDdown2(layer2_C)

        # Cross Channel Module
        B, C, H, W = layer1_A.size()
        layer1_D = torch.empty(B, C * 2, H, W).cuda()
        layer1_D[:, ::2, :, :] = layer1_A
        layer1_D[:, 1::2, :, :] = layer1_B
        combined_layer1 = torch.cat((layer1_A, layer1_B, layer1_D), dim=1)

        B, C, H, W = layer2_A.size()
        layer2_D = torch.empty(B, C * 2, H, W).cuda()
        layer2_D[:, ::2, :, :] = layer2_A
        layer2_D[:, 1::2, :, :] = layer2_B
        combined_layer2 = torch.cat((layer2_A, layer2_B, layer2_D), dim=1)

        B, C, H, W = layer3_A.size()
        layer3_D = torch.empty(B, C * 2, H, W).cuda()
        layer3_D[:, ::2, :, :] = layer3_A
        layer3_D[:, 1::2, :, :] = layer3_B
        combined_layer3 = torch.cat((layer3_A, layer3_B, layer3_D), dim=1)
        combined_layer1 = self.conv_reduce_1(combined_layer1)
        combined_layer2 = self.conv_reduce_2(combined_layer2)
        combined_layer3 = self.conv_reduce_3(combined_layer3)

        # Change Guide Module
        change_map1 = self.CDdecoder1(layer1_C)
        change_map2 = self.CDdecoder2(layer2_C)
        change_map3 = self.CDdecoder3(layer3_C)

        layer3_out = self.cgm_3(combined_layer3, change_map3)
        feature3 = self.decoder_module3(
            torch.cat([self.upsample2x(layer3_out), combined_layer2], 1)
        )
        layer2_out = self.cgm_2(feature3, change_map2)
        feature2 = self.decoder_module2(
            torch.cat([self.upsample2x(layer2_out), combined_layer1], 1)
        )
        layer1_out = self.cgm_1(feature2, change_map1)

        final_map = self.decoder_final(layer1_out)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)
        change_maps3 = F.interpolate(
            change_map3, size, mode="bilinear", align_corners=True
        )
        change_maps2 = F.interpolate(
            change_map2, size, mode="bilinear", align_corners=True
        )
        change_maps1 = F.interpolate(
            change_map1, size, mode="bilinear", align_corners=True
        )

        return change_maps3, change_maps2, change_maps1, final_map


class N_BPMSNet_ST(nn.Module):
    def __init__(self):
        super(N_BPMSNet_ST, self).__init__()

        # Smaller backbone: Use MobileNetV2 as a lightweight feature extractor
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.load_state_dict(torch.load("./network/mobilenet_v2-b0353104.pth"))
        self.features = copy.deepcopy(mobilenet.features[:7])
        self.features_1 = copy.deepcopy(mobilenet.features[:7])
        old_conv = self.features[0][0]
        new_conv = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
        self.features[0][0] = new_conv
        self.features_1[0][0] = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.features_1[0][0].weight = old_conv.weight

        # Downsampled feature layers
        self.down1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.CDdown1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.CDdown2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_reduce_1 = BasicConv2d(32 * 4, 32, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 4, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(128 * 4, 128, 3, 1, 1)

        self.conv_reduce_4 = BasicConv2d(32 * 5, 32, 3, 1, 1)
        self.conv_reduce_5 = BasicConv2d(64 * 5, 64, 3, 1, 1)
        self.conv_reduce_6 = BasicConv2d(128 * 5, 128, 3, 1, 1)

        self.st_block_1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.st_block_2 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.st_block_3 = BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.CDdecoder1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.CDdecoder2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.CDdecoder3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.decoder_module3 = BasicConv2d(192, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(96, 32, 3, 1, 1)
        self.decoder_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

        self.cgm_1 = ChangeGuideModule(32)
        self.cgm_2 = ChangeGuideModule(64)
        self.cgm_3 = ChangeGuideModule(128)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B, C):
        size = A.size()[2:]

        layer1_A = self.features(A)
        layer1_B = self.features(B)
        layer1_C = self.features_1(C)

        layer2_A = self.down1(layer1_A)
        layer3_A = self.down2(layer2_A)

        layer2_B = self.down1(layer1_B)
        layer3_B = self.down2(layer2_B)

        layer2_C = self.CDdown1(layer1_C)
        layer3_C = self.CDdown2(layer2_C)

        B, C, H, W = layer1_A.size()
        layer1_D = torch.empty(B, C * 2, H, W).cuda()
        layer1_D[:, ::2, :, :] = layer1_A
        layer1_D[:, 1::2, :, :] = layer1_B
        combined_layer1 = torch.cat((layer1_A, layer1_B, layer1_D), dim=1)

        B, C, H, W = layer2_A.size()
        layer2_D = torch.empty(B, C * 2, H, W).cuda()
        layer2_D[:, ::2, :, :] = layer2_A
        layer2_D[:, 1::2, :, :] = layer2_B
        combined_layer2 = torch.cat((layer2_A, layer2_B, layer2_D), dim=1)

        B, C, H, W = layer3_A.size()
        layer3_D = torch.empty(B, C * 2, H, W).cuda()
        layer3_D[:, ::2, :, :] = layer3_A
        layer3_D[:, 1::2, :, :] = layer3_B
        combined_layer3 = torch.cat((layer3_A, layer3_B, layer3_D), dim=1)

        combined_layer1 = self.conv_reduce_1(combined_layer1)
        combined_layer2 = self.conv_reduce_2(combined_layer2)
        combined_layer3 = self.conv_reduce_3(combined_layer3)

        change_map3 = self.CDdecoder3(layer3_C)

        layer3_out_1 = self.cgm_3(
            combined_layer3, change_map3
        )  # Adjust based on the changed map
        B, C, H, W = combined_layer3.size()
        ct_tensor_32 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_32[:, :, :, ::2] = layer3_A
        ct_tensor_32[:, :, :, 1::2] = layer3_B

        layer3_out_2 = self.st_block_3(ct_tensor_32)
        ct_tensor_33 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_33[:, :, :, 0:W] = layer3_A
        ct_tensor_33[:, :, :, W:] = layer3_B

        layer3_out_3 = self.st_block_3(ct_tensor_33)
        layer3_out = self.conv_reduce_6(
            torch.cat(
                [
                    layer3_out_1,
                    layer3_out_2[:, :, :, ::2],
                    layer3_out_2[:, :, :, 1::2],
                    layer3_out_3[:, :, :, 0:W],
                    layer3_out_3[:, :, :, W:],
                ],
                dim=1,
            )
        )
        feature3 = self.decoder_module3(
            torch.cat([self.upsample2x(layer3_out), combined_layer2], 1)
        )

        layer2_out_1 = self.cgm_2(feature3, change_map3)
        B, C, H, W = feature3.size()
        ct_tensor_22 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_22[:, :, :, ::2] = layer2_A
        ct_tensor_22[:, :, :, 1::2] = layer2_B
        layer2_out_2 = self.st_block_2(ct_tensor_22)
        ct_tensor_22 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_22[:, :, :, 0:W] = layer2_A
        ct_tensor_22[:, :, :, W:] = layer2_B
        layer2_out_3 = self.st_block_2(ct_tensor_22)
        layer2_out = self.conv_reduce_5(
            torch.cat(
                [
                    layer2_out_1,
                    layer2_out_2[:, :, :, ::2],
                    layer2_out_2[:, :, :, 1::2],
                    layer2_out_3[:, :, :, 0:W],
                    layer2_out_3[:, :, :, W:],
                ],
                dim=1,
            )
        )
        feature2 = self.decoder_module2(
            torch.cat([self.upsample2x(layer2_out), combined_layer1], 1)
        )

        layer1_out_1 = self.cgm_1(feature2, change_map3)
        B, C, H, W = feature2.size()
        ct_tensor_12 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_12[:, :, :, ::2] = layer1_A
        ct_tensor_12[:, :, :, 1::2] = layer1_B
        layer1_out_2 = self.st_block_1(ct_tensor_12)
        ct_tensor_12 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_12[:, :, :, 0:W] = layer1_A
        ct_tensor_12[:, :, :, W:] = layer1_B
        layer1_out_3 = self.st_block_1(ct_tensor_12)
        layer1_out = self.conv_reduce_4(
            torch.cat(
                [
                    layer1_out_1,
                    layer1_out_2[:, :, :, ::2],
                    layer1_out_2[:, :, :, 1::2],
                    layer1_out_3[:, :, :, 0:W],
                    layer1_out_3[:, :, :, W:],
                ],
                dim=1,
            )
        )

        final_map = self.decoder_final(layer1_out)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)
        change_maps3 = F.interpolate(
            change_map3, size, mode="bilinear", align_corners=True
        )

        return change_maps3, None, None, final_map


class N_BPMSNet_VSS(nn.Module):
    def __init__(self):
        super(N_BPMSNet_VSS, self).__init__()

        # Smaller backbone: Use MobileNetV2 as a lightweight feature extractor
        mobilenet = models.mobilenet_v2(pretrained=False)
        mobilenet.load_state_dict(torch.load("./network/mobilenet_v2-b0353104.pth"))
        self.features = copy.deepcopy(mobilenet.features[:7])
        self.features_1 = copy.deepcopy(mobilenet.features[:7])
        old_conv = self.features[0][0]
        new_conv = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            if new_conv.in_channels > 3:
                new_conv.weight[:, 3:, :, :] = torch.mean(
                    old_conv.weight, dim=1, keepdim=True
                )
        self.features[0][0] = new_conv
        self.features_1[0][0] = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.features_1[0][0].weight = old_conv.weight

        # Downsampled feature layers
        self.drop_path = DropPath(0.1)
        self.mlp1 = Mlp(
            in_features=96,
            hidden_features=96 * 4,
            act_layer=nn.GELU,
            drop=0.0,
            channels_first=False,
        )  # CHANGE!!!
        self.norm1 = nn.LayerNorm(96)
        self.op1 = SS2D(
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
        )
        self.down1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.mlp2 = Mlp(
            in_features=96,
            hidden_features=96 * 4,
            act_layer=nn.GELU,
            drop=0.0,
            channels_first=False,
        )  # CHANGE!!!
        self.norm2 = nn.LayerNorm(96)
        self.op2 = SS2D(
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            initialize="v0",
            forward_type="v2",
            channel_first=False,
        )
        self.down2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.CDdown1 = BasicConv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )  # CHANGE !!!!!!
        self.CDdown2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.conv_reduce_1 = BasicConv2d(32 * 2, 32, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(64 * 2, 64, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(128 * 2, 128, 3, 1, 1)

        self.CDdecoder3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.decoder_module3 = BasicConv2d(192, 64, 3, 1, 1)
        self.decoder_module2 = BasicConv2d(96, 32, 3, 1, 1)
        self.decoder_final = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1))

        self.cgm_1 = ChangeGuideModule(32)
        self.cgm_2 = ChangeGuideModule(64)
        self.cgm_3 = ChangeGuideModule(128)

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B, C):
        size = A.size()[2:]

        layer1_A = self.features(A)
        layer1_B = self.features(B)
        layer1_C = self.features_1(C)

        layer2_A = self.down1(layer1_A)
        layer2_A = layer2_A + self.drop_path(
            self.op1(self.norm1(layer2_A))
        )  # CHANGE!!!
        layer2_A = layer2_A + self.drop_path(self.mlp1(self.norm1(layer2_A)))

        layer3_A = self.down2(layer2_A)
        layer3_A = layer3_A + self.drop_path(self.op2(self.norm2(layer3_A)))
        layer3_A = layer3_A + self.drop_path(self.mlp2(self.norm2(layer3_A)))

        layer2_B = self.down1(layer1_B)
        layer2_B = layer2_B + self.drop_path(self.op1(self.norm1(layer2_B)))
        layer2_B = layer2_B + self.drop_path(self.mlp1(self.norm1(layer2_B)))

        layer3_B = self.down2(layer2_B)
        layer3_B = layer3_B + self.drop_path(self.op2(self.norm2(layer3_B)))
        layer3_B = layer3_B + self.drop_path(self.mlp2(self.norm2(layer3_B)))

        layer2_C = self.CDdown1(layer1_C)
        layer3_C = self.CDdown2(layer2_C)

        # p31 = self.st_block_31(torch.cat([layer3_A, layer3_B], dim=1))
        B, C, H, W = layer3_A.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = layer3_A  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = layer3_B  # Even columns
        # p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_33[:, :, :, 0:W] = layer3_A
        ct_tensor_33[:, :, :, W:] = layer3_B
        # p33 = self.st_block_33(ct_tensor_33)

        # p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        # p3 = self._upsample_add(p4, p3)
        # p3 = self.smooth_layer_3(p3)

        combined_layer1 = torch.cat((layer1_A, layer1_B), dim=1)
        combined_layer2 = torch.cat((layer2_A, layer2_B), dim=1)
        combined_layer3 = torch.cat((layer3_A, layer3_B), dim=1)
        combined_layer1 = self.conv_reduce_1(combined_layer1)
        combined_layer2 = self.conv_reduce_2(combined_layer2)
        combined_layer3 = self.conv_reduce_3(combined_layer3)

        change_map3 = self.CDdecoder3(layer3_C)

        layer3_out = self.cgm_3(combined_layer3, change_map3)
        feature3 = self.decoder_module3(
            torch.cat([self.upsample2x(layer3_out), combined_layer2], 1)
        )
        layer2_out = self.cgm_2(feature3, change_map3)
        feature2 = self.decoder_module2(
            torch.cat([self.upsample2x(layer2_out), combined_layer1], 1)
        )
        layer1_out = self.cgm_1(feature2, change_map3)

        final_map = self.decoder_final(layer1_out)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)
        change_maps4 = F.interpolate(
            change_map3, size, mode="bilinear", align_corners=True
        )

        return change_maps4, final_map


if __name__ == "__main__":
    # 测试热图
    # net=CGNet().cuda()
    # out=net(torch.rand((2,3,256,256)).cuda(),torch.rand((2,3,256,256)).cuda())

    # 测试模型大小
    print("Hi~")
    input_size = 256
    model_restoration = N_BPMSNet()
    from ptflops import get_model_complexity_info
    from torchstat import stat

    # input = torch.rand((3, input_size, input_size))
    # output = model_restoration(input)
    macs, params = get_model_complexity_info(
        model_restoration,
        (3, input_size, input_size),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )
    stat(model_restoration, (3, input_size, input_size))
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
