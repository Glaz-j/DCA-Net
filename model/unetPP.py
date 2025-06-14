import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class NestedUNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, deep_supervision=False, fix_size=256):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.deep_supervision = deep_supervision
        self.fix_size = fix_size  # 固定处理尺寸（设为None则不调整）

        nb_filters = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 编码器部分
        self.conv0_0 = VGGBlock(n_channels, nb_filters[0], nb_filters[0])
        self.conv1_0 = VGGBlock(nb_filters[0], nb_filters[1], nb_filters[1])
        self.conv2_0 = VGGBlock(nb_filters[1], nb_filters[2], nb_filters[2])
        self.conv3_0 = VGGBlock(nb_filters[2], nb_filters[3], nb_filters[3])
        self.conv4_0 = VGGBlock(nb_filters[3], nb_filters[4], nb_filters[4])

        # 跳跃连接部分
        self.conv0_1 = VGGBlock(nb_filters[0] + nb_filters[1], nb_filters[0], nb_filters[0])
        self.conv1_1 = VGGBlock(nb_filters[1] + nb_filters[2], nb_filters[1], nb_filters[1])
        self.conv2_1 = VGGBlock(nb_filters[2] + nb_filters[3], nb_filters[2], nb_filters[2])
        self.conv3_1 = VGGBlock(nb_filters[3] + nb_filters[4], nb_filters[3], nb_filters[3])

        self.conv0_2 = VGGBlock(nb_filters[0] * 2 + nb_filters[1], nb_filters[0], nb_filters[0])
        self.conv1_2 = VGGBlock(nb_filters[1] * 2 + nb_filters[2], nb_filters[1], nb_filters[1])
        self.conv2_2 = VGGBlock(nb_filters[2] * 2 + nb_filters[3], nb_filters[2], nb_filters[2])

        self.conv0_3 = VGGBlock(nb_filters[0] * 3 + nb_filters[1], nb_filters[0], nb_filters[0])
        self.conv1_3 = VGGBlock(nb_filters[1] * 3 + nb_filters[2], nb_filters[1], nb_filters[1])

        self.conv0_4 = VGGBlock(nb_filters[0] * 4 + nb_filters[1], nb_filters[0], nb_filters[0])

        # 输出层
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filters[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filters[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filters[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filters[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filters[0], n_classes, kernel_size=1)

    def _resize_tensor(self, x, size):
        """统一调整张量尺寸"""
        if size is None:
            return x
        return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)

    def _restore_size(self, x, original_size):
        """还原到原始尺寸"""
        return F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 记录原始尺寸 [N, C, H, W]
        original_size = x.shape[2:]

        # Step 1: 统一调整输入尺寸
        if self.fix_size is not None:
            x = self._resize_tensor(x, self.fix_size)

        # 编码器路径
        x0_0 = self.conv0_0(x)  # [N, 32, H, W]
        x1_0 = self.conv1_0(self.pool(x0_0))  # [N, 64, H/2, W/2]
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))  # [N, 32, H, W]

        x2_0 = self.conv2_0(self.pool(x1_0))  # [N, 128, H/4, W/4]
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))  # [N, 64, H/2, W/2]
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))  # [N, 32, H, W]

        x3_0 = self.conv3_0(self.pool(x2_0))  # [N, 256, H/8, W/8]
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))  # [N, 128, H/4, W/4]
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))  # [N, 64, H/2, W/2]
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))  # [N, 32, H, W]

        x4_0 = self.conv4_0(self.pool(x3_0))  # [N, 512, H/16, W/16]
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))  # [N, 256, H/8, W/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))  # [N, 128, H/4, W/4]
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))  # [N, 64, H/2, W/2]
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))  # [N, 32, H, W]

        # Step 2: 还原到原始尺寸
        if self.deep_supervision:
            out1 = self._restore_size(self.final1(x0_1), original_size)
            out2 = self._restore_size(self.final2(x0_2), original_size)
            out3 = self._restore_size(self.final3(x0_3), original_size)
            out4 = self._restore_size(self.final4(x0_4), original_size)
            return [out1, out2, out3, out4]
        else:
            out = self.final(x0_4)
            return self._restore_size(out, original_size)


