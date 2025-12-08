# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(16, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(16, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SELayer(nn.Module):
    def __init__(self, channel_rgb, channel_hsi, reduction=16):
        super(SELayer, self).__init__()
        self.cross_attention_rgb = nn.MultiheadAttention(embed_dim=channel_rgb, num_heads=4)
        self.cross_attention_hsi = nn.MultiheadAttention(embed_dim=channel_hsi, num_heads=4)
        self.spatial_attention_rgb = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.spatial_attention_hsi = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.avg_pool_rgb = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_hsi = nn.AdaptiveAvgPool2d(1)
        self.fc_rgb = nn.Sequential(
            nn.Linear(channel_rgb, channel_rgb // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_rgb // reduction, channel_rgb, bias=False),
            nn.Sigmoid()
        )
        self.fc_hsi = nn.Sequential(
            nn.Linear(channel_hsi, channel_hsi // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_hsi // reduction, channel_hsi, bias=False),
            nn.Sigmoid()
        )
        self.l2_reg_weight = 1e-4

    def forward(self, rgb, hsi):
        print(f"SELayer input RGB shape: {rgb.shape}")
        print(f"SELayer input HSI shape: {hsi.shape}")
        b, c_rgb, h, w = rgb.size()
        rgb_max_pool, _ = torch.max(rgb, dim=1, keepdim=True)
        rgb_avg_pool = torch.mean(rgb, dim=1, keepdim=True)
        rgb_spatial = torch.cat([rgb_max_pool, rgb_avg_pool], dim=1)
        rgb_spatial_weight = self.spatial_attention_rgb(rgb_spatial)
        rgb_spatial_attended = rgb * rgb_spatial_weight
        b, c_hsi, h, w = hsi.size()
        hsi_max_pool, _ = torch.max(hsi, dim=1, keepdim=True)
        hsi_avg_pool = torch.mean(hsi, dim=1, keepdim=True)
        hsi_spatial = torch.cat([hsi_max_pool, hsi_avg_pool], dim=1)
        hsi_spatial_weight = self.spatial_attention_hsi(hsi_spatial)
        hsi_spatial_attended = hsi * hsi_spatial_weight
        rgb_flat = rgb_spatial_attended.view(b, c_rgb, -1).permute(2, 0, 1)
        hsi_flat = hsi_spatial_attended.view(b, c_hsi, -1).permute(2, 0, 1)
        rgb_attended, rgb_attn_weights = self.cross_attention_rgb(rgb_flat, hsi_flat, hsi_flat)
        rgb_attended = rgb_attended.permute(1, 2, 0).view(b, c_rgb, h, w)
        hsi_attended, hsi_attn_weights = self.cross_attention_hsi(hsi_flat, rgb_flat, rgb_flat)
        hsi_attended = hsi_attended.permute(1, 2, 0).view(b, c_hsi, h, w)
        rgb_se = self.avg_pool_rgb(rgb_attended).view(b, c_rgb)
        rgb_se = self.fc_rgb(rgb_se).view(b, c_rgb, 1, 1)
        rgb_se = rgb_attended * rgb_se
        hsi_se = self.avg_pool_hsi(hsi_attended).view(b, c_hsi)
        hsi_se = self.fc_hsi(hsi_se).view(b, c_hsi, 1, 1)
        hsi_se = hsi_attended * hsi_se
        l2_reg_loss = 0.0
        l2_reg_loss += self.l2_reg_weight * torch.norm(rgb_spatial_weight, p=2)
        l2_reg_loss += self.l2_reg_weight * torch.norm(hsi_spatial_weight, p=2)
        l2_reg_loss += self.l2_reg_weight * torch.norm(rgb_attn_weights, p=2)
        l2_reg_loss += self.l2_reg_weight * torch.norm(hsi_attn_weights, p=2)
        l2_reg_loss += self.l2_reg_weight * torch.norm(rgb_se, p=2)
        l2_reg_loss += self.l2_reg_weight * torch.norm(hsi_se, p=2)
        fused = rgb_se + hsi_se
        print(f"SELayer output shape: {fused.shape}")
        return fused, l2_reg_loss


class FuseResNetV2(nn.Module):
    def __init__(self, block_units, width_factor, num_classes=2, in_channels_hsi=64):
        super(FuseResNetV2, self).__init__()
        if len(block_units) != 4:
            raise ValueError(f"block_units must be a tuple of length 4, got {block_units}")
        width = int(32 * width_factor)
        self.width = width
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
        self.se_layer0 = SELayer(channel_rgb=width, channel_hsi=width)
        self.se_layer1 = SELayer(channel_rgb=width, channel_hsi=width)
        self.se_layer2 = SELayer(channel_rgb=width * 2, channel_hsi=width * 2)
        self.se_layer3 = SELayer(channel_rgb=width * 4, channel_hsi=width * 4)
        self.se_layer4 = SELayer(channel_rgb=width * 8, channel_hsi=width * 8)
        self.body1 = self._make_layer(BasicBlock, width, block_units[0], stride=1)
        self.body2 = self._make_layer(BasicBlock, width * 2, block_units[1], stride=2)
        self.body3 = self._make_layer(BasicBlock, width * 4, block_units[2], stride=2)
        self.body4 = self._make_layer(BasicBlock, width * 8, block_units[3], stride=1)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(16, width * 2)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 4, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(16, width * 4)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(width * 4, width * 4, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(16, width * 4)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.width != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.width, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(16, planes * block.expansion),
            )
        layers = []
        layers.append(block(self.width, planes, stride, downsample))
        self.width = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.width, planes))
        return nn.Sequential(*layers)

    def forward(self, rgb, hsi):
        print(f"FuseResNetV2 input RGB shape: {rgb.shape}")
        print(f"FuseResNetV2 input HSI shape: {hsi.shape}")
        rgb = self.root_rgb(rgb)
        hsi = self.root_hsi(hsi)
        print(f"root_rgb output shape: {rgb.shape}")
        print(f"root_hsi output shape: {hsi.shape}")
        hsi = nn.functional.interpolate(hsi, size=rgb.shape[2:], mode='bilinear', align_corners=False)
        print(f"Interpolated HSI shape: {hsi.shape}")
        fused0, reg_loss0 = self.se_layer0(rgb, hsi)
        print(f"SELayer0 output shape: {fused0.shape}")
        fused1 = self.body1(fused0)
        print(f"body1 output shape: {fused1.shape}")
        fused1, reg_loss1 = self.se_layer1(fused1, fused1)
        print(f"SELayer1 output shape: {fused1.shape}")
        fused2 = self.body2(fused1)
        print(f"body2 output shape: {fused2.shape}")
        fused2, reg_loss2 = self.se_layer2(fused2, fused2)
        print(f"SELayer2 output shape: {fused2.shape}")
        fused3 = self.body3(fused2)
        print(f"body3 output shape: {fused3.shape}")
        fused3, reg_loss3 = self.se_layer3(fused3, fused3)
        print(f"SELayer3 output shape: {fused3.shape}")
        fused4 = self.body4(fused3)
        print(f"body4 output shape: {fused4.shape}")
        fused4, reg_loss4 = self.se_layer4(fused4, fused4)
        print(f"SELayer4 output shape: {fused4.shape}")
        skip1 = self.downsample1(fused1)
        skip2 = self.downsample2(fused2)
        skip3 = self.downsample3(fused3)
        print(f"skip1 shape: {skip1.shape}")
        print(f"skip2 shape: {skip2.shape}")
        print(f"skip3 shape: {skip3.shape}")
        total_reg_loss = reg_loss0 + reg_loss1 + reg_loss2 + reg_loss3 + reg_loss4
        return fused4, total_reg_loss, [fused4, skip3, skip2, skip1]

    def load_pretrained_weights(self, weights, freeze=False):
        state_dict = self.state_dict()
        for name, param in weights.items():
            if name in state_dict:
                param = torch.from_numpy(param).float()
                if param.shape == state_dict[name].shape:
                    state_dict[name].copy_(param)
                else:
                    print(f"Skipping {name} due to shape mismatch: {param.shape} vs {state_dict[name].shape}")
        if freeze:
            for param in self.parameters():
                param.requires_grad = False