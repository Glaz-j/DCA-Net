import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # 引入F模块，用于interpolate


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), # 也可以使用 F.interpolate
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttU_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, scale_factor=1):
        super(AttU_Net, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 记录原始输入图像的 H 和 W
        original_H = x.shape[2]
        original_W = x.shape[3]

        # 将输入图像调整为 256x256
        # 注意：interpolate 的输入是 (N, C, H, W)
        x_resized = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # encoding path
        x1 = self.Conv1(x_resized) # 使用调整后的图像进行后续操作

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        # 注意：这里需要确保 x4 的尺寸与 d5 匹配，或者在 Att5 内部处理
        # 由于 AttU-Net 的设计，跳跃连接的特征图 x1, x2, x3, x4 的尺寸会随着下采样而减小
        # 而 d5 是上采样后的，理论上 Att5 的输入 g 和 x 应该有相同的 H 和 W
        # 如果你的原始 AttU-Net 实现是正确的，这里应该不需要额外调整 x4 的尺寸
        # 但如果输入 Att5 的 g 和 x 尺寸不一致，你需要检查 Att5 的设计或在上采样/下采样过程中确保尺寸匹配
        # 为了保险起见，可以对 x4 (来自编码路径，尺寸可能不是 d5 的尺寸) 进行插值以匹配 d5
        x4_resized_for_att = F.interpolate(x4, size=(d5.shape[2], d5.shape[3]), mode='bilinear', align_corners=False)
        g_d5 = self.Att5(g=d5, x=x4_resized_for_att) # 使用调整尺寸后的 x4
        d5 = torch.cat((g_d5, d5), dim=1) # 注意这里应该是 g_d5 而不是 x4
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3_resized_for_att = F.interpolate(x3, size=(d4.shape[2], d4.shape[3]), mode='bilinear', align_corners=False)
        g_d4 = self.Att4(g=d4, x=x3_resized_for_att)
        d4 = torch.cat((g_d4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_resized_for_att = F.interpolate(x2, size=(d3.shape[2], d3.shape[3]), mode='bilinear', align_corners=False)
        g_d3 = self.Att3(g=d3, x=x2_resized_for_att)
        d3 = torch.cat((g_d3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_resized_for_att = F.interpolate(x1, size=(d2.shape[2], d2.shape[3]), mode='bilinear', align_corners=False)
        g_d2 = self.Att2(g=d2, x=x1_resized_for_att)
        d2 = torch.cat((g_d2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # 将输出恢复到原始图像尺寸
        output = F.interpolate(d1, size=(original_H, original_W), mode='bilinear', align_corners=False)

        return output