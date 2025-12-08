# vit_model/vit_transformer/hybrid_resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(16, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(16, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        elif out.shape != identity.shape:
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(out + identity)


class SELayer(nn.Module):
    def __init__(self, channel_rgb, channel_hsi, reduction=16):
        super().__init__()
        self.avg_pool_rgb = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_hsi = nn.AdaptiveAvgPool2d(1)
        self.fc_rgb = nn.Sequential(
            nn.Linear(channel_rgb, channel_rgb // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel_rgb // reduction, channel_rgb),
            nn.Sigmoid()
        )
        self.fc_hsi = nn.Sequential(
            nn.Linear(channel_hsi, channel_hsi // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hsi // reduction, channel_hsi),
            nn.Sigmoid()
        )

    def forward(self, rgb, hsi):
        b, c, _, _ = rgb.size()
        rgb_se = self.avg_pool_rgb(rgb).view(b, c)
        rgb_se = self.fc_rgb(rgb_se).view(b, c, 1, 1)
        rgb = rgb * rgb_se

        b, c, _, _ = hsi.size()
        hsi_se = self.avg_pool_hsi(hsi).view(b, c)
        hsi_se = self.fc_hsi(hsi_se).view(b, c, 1, 1)
        hsi = hsi * hsi_se

        if rgb.shape != hsi.shape:
            hsi = F.interpolate(hsi, size=rgb.shape[2:], mode='bilinear', align_corners=False)

        return rgb + hsi, 0.0  # 返回融合特征和正则项（此处为0）


class FuseResNetV2(nn.Module):
    def __init__(self, block_units, width_factor, num_classes=2, in_channels_hsi=60):
        super().__init__()
        width = int(32 * width_factor)
        self.inplanes = width  # 初始输入通道数
        self.root_rgb = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GroupNorm(16, width),
            nn.ReLU(inplace=True)
        )
        self.root_hsi = nn.Sequential(
            nn.Conv2d(in_channels_hsi, width, kernel_size=7, stride=4, padding=3, bias=False),
            nn.GroupNorm(16, width),
            nn.ReLU(inplace=True)
        )
        self.se_layer0 = SELayer(width, width)
        self.body1 = self._make_layer(BasicBlock, width, block_units[0], stride=1)
        self.body2 = self._make_layer(BasicBlock, width * 2, block_units[1], stride=2)
        self.body3 = self._make_layer(BasicBlock, width * 4, block_units[2], stride=2)
        self.body4 = self._make_layer(BasicBlock, width * 8, block_units[3], stride=1)

        self.downsample1 = nn.Conv2d(width, width * 2, kernel_size=1)
        self.downsample2 = nn.Conv2d(width * 2, width * 4, kernel_size=1)
        self.downsample3 = nn.Conv2d(width * 4, width * 8, kernel_size=1)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(16, planes * block.expansion)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, rgb, hsi):
        rgb = self.root_rgb(rgb)
        hsi = self.root_hsi(hsi)
        fused0, reg_loss0 = self.se_layer0(rgb, hsi)
        fused1 = self.body1(fused0)
        fused2 = self.body2(fused1)
        fused3 = self.body3(fused2)
        fused4 = self.body4(fused3)
        skip1 = self.downsample1(fused1)
        skip2 = self.downsample2(fused2)
        skip3 = self.downsample3(fused3)
        return fused4, reg_loss0, [fused4, skip3, skip2, skip1]

