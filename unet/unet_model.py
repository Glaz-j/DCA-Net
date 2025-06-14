""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .coordatt import *
from utils.utils import *

class UNetCoordAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        #简单的在unet上增加CroodAttention
        super(UNetCoordAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.coord_att = CoordAtt(inp=1024, oup=1024)

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.coord_att(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class DoubleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DoubleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.cartesian_to_polar = cartesian_to_polar#  笛卡尔系转极坐标系
        self.polar_to_cartesian = polar_to_cartesian # 极坐标系转笛卡尔坐标系


        self.inc_cartesian = (DoubleConv(n_channels, 64))
        self.down_cartesian1 = (Down(64, 128))
        self.down_cartesian2 = (Down(128, 256))
        self.down_cartesian3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_cartesian4 = (Down(512, 1024 // factor))
        self.up_cartesian1 = (Up(1024, 512 // factor, bilinear))
        self.up_cartesian2 = (Up(512, 256 // factor, bilinear))
        self.up_cartesian3 = (Up(256, 128 // factor, bilinear))
        self.up_cartesian4 = (Up(128, 64, bilinear))
        self.out_cartesian_c = (OutConv(64, n_classes))

        self.inc_polar = (DoubleConv(n_channels, 64))
        self.down_polar1 = (Down(64, 128))
        self.down_polar2 = (Down(128, 256))
        self.down_polar3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_polar4 = (Down(512, 1024 // factor))
        self.up_polar1 = (Up(1024, 512 // factor, bilinear))
        self.up_polar2 = (Up(512, 256 // factor, bilinear))
        self.up_polar3 = (Up(256, 128 // factor, bilinear))
        self.up_polar4 = (Up(128, 64, bilinear))
        self.out_polar_c = (OutConv(64, n_classes))

        self.merge = nn.Conv2d(2*n_classes, n_classes, kernel_size=3, padding=1) #合并两个特征图


    def forward(self, x):
        x_cartesian = x
        x_polar = x

        #笛卡尔坐标系
        x1_cartesian = self.inc_cartesian(x_cartesian)
        x2_cartesian = self.down_cartesian1(x1_cartesian)
        x3_cartesian = self.down_cartesian2(x2_cartesian)
        x4_cartesian = self.down_cartesian3(x3_cartesian)
        x5_cartesian = self.down_cartesian4(x4_cartesian)
        x = self.up_cartesian1(x5_cartesian, x4_cartesian)
        x = self.up_cartesian2(x, x3_cartesian)
        x = self.up_cartesian3(x, x2_cartesian)
        x = self.up_cartesian4(x, x1_cartesian)
        logits_cartesian = self.out_cartesian_c(x)

        #极坐标坐标系
        x_polar = self.cartesian_to_polar(x_polar)
        x1_polar = self.inc_polar(x_polar)
        x2_polar = self.down_polar1(x1_polar)
        x3_polar = self.down_polar2(x2_polar)
        x4_polar = self.down_polar3(x3_polar)
        x5_polar = self.down_polar4(x4_polar)
        x = self.up_polar1(x5_polar, x4_polar)
        x = self.up_polar2(x, x3_polar)
        x = self.up_polar3(x, x2_polar)
        x = self.up_polar4(x, x1_polar)
        logits_polar = self.out_polar_c(x)
        logits_polar = self.polar_to_cartesian(logits_polar,logits_polar.shape[2],logits_polar.shape[3])

        logits = torch.cat([logits_cartesian, logits_polar], dim=1)  # Shape: [B, 2C, H, W]

        mask = self.merge(logits)


        return mask

    def use_checkpointing(self):
        """启用双路径检查点机制，节省显存"""
        # 笛卡尔路径编码器
        self.inc_cartesian = torch.utils.checkpoint(self.inc_cartesian)
        self.down_cartesian1 = torch.utils.checkpoint(self.down_cartesian1)
        self.down_cartesian2 = torch.utils.checkpoint(self.down_cartesian2)
        self.down_cartesian3 = torch.utils.checkpoint(self.down_cartesian3)
        self.down_cartesian4 = torch.utils.checkpoint(self.down_cartesian4)

        # 笛卡尔路径解码器
        self.up_cartesian1 = torch.utils.checkpoint(self.up_cartesian1)
        self.up_cartesian2 = torch.utils.checkpoint(self.up_cartesian2)
        self.up_cartesian3 = torch.utils.checkpoint(self.up_cartesian3)
        self.up_cartesian4 = torch.utils.checkpoint(self.up_cartesian4)
        self.out_cartesian_c = torch.utils.checkpoint(self.out_cartesian_c)

        # 极坐标路径编码器
        self.inc_polar = torch.utils.checkpoint(self.inc_polar)
        self.down_polar1 = torch.utils.checkpoint(self.down_polar1)
        self.down_polar2 = torch.utils.checkpoint(self.down_polar2)
        self.down_polar3 = torch.utils.checkpoint(self.down_polar3)
        self.down_polar4 = torch.utils.checkpoint(self.down_polar4)

        # 极坐标路径解码器
        self.up_polar1 = torch.utils.checkpoint(self.up_polar1)
        self.up_polar2 = torch.utils.checkpoint(self.up_polar2)
        self.up_polar3 = torch.utils.checkpoint(self.up_polar3)
        self.up_polar4 = torch.utils.checkpoint(self.up_polar4)
        self.out_polar_c = torch.utils.checkpoint(self.out_polar_c)


class DoubleCoordAttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DoubleCoordAttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.cartesian_to_polar = cartesian_to_polar#  笛卡尔系转极坐标系
        self.polar_to_cartesian = polar_to_cartesian # 极坐标系转笛卡尔坐标系


        self.inc_cartesian = (DoubleConv(n_channels, 64))
        self.down_cartesian1 = (Down(64, 128))
        self.down_cartesian2 = (Down(128, 256))
        self.down_cartesian3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_cartesian4 = (Down(512, 1024 // factor))
        self.up_cartesian1 = (Up(1024, 512 // factor, bilinear))
        self.up_cartesian2 = (Up(512, 256 // factor, bilinear))
        self.up_cartesian3 = (Up(256, 128 // factor, bilinear))
        self.up_cartesian4 = (Up(128, 64, bilinear))
        self.out_cartesian_c = (OutConv(64, n_classes))

        self.inc_polar = (DoubleConv(n_channels, 64))
        self.down_polar1 = (Down(64, 128))
        self.down_polar2 = (Down(128, 256))
        self.down_polar3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_polar4 = (Down(512, 1024 // factor))
        self.up_polar1 = (Up(1024, 512 // factor, bilinear))
        self.up_polar2 = (Up(512, 256 // factor, bilinear))
        self.up_polar3 = (Up(256, 128 // factor, bilinear))
        self.up_polar4 = (Up(128, 64, bilinear))
        self.out_polar_c = (OutConv(64, n_classes))

        #双特征图注意力
        self.coord_att = DoubleCoordAtt(inp=1024, oup=1024)

        self.merge = nn.Conv2d(2*n_classes, n_classes, kernel_size=3, padding=1) #合并两个特征图


    def forward(self, x):
        x_cartesian = x
        x_polar = x

        #笛卡尔坐标系下采样
        x1_cartesian = self.inc_cartesian(x_cartesian)
        x2_cartesian = self.down_cartesian1(x1_cartesian)
        x3_cartesian = self.down_cartesian2(x2_cartesian)
        x4_cartesian = self.down_cartesian3(x3_cartesian)
        x5_cartesian = self.down_cartesian4(x4_cartesian)

        # 极坐标坐标系下采样
        x_polar = self.cartesian_to_polar(x_polar)
        x1_polar = self.inc_polar(x_polar)
        x2_polar = self.down_polar1(x1_polar)
        x3_polar = self.down_polar2(x2_polar)
        x4_polar = self.down_polar3(x3_polar)
        x5_polar = self.down_polar4(x4_polar)

        x5_cartesian, x5_polar = self.coord_att(x5_cartesian, x5_polar)

        # 笛卡尔坐标系上采样
        x_cartesian = self.up_cartesian1(x5_cartesian, x4_cartesian)
        x_cartesian = self.up_cartesian2(x_cartesian, x3_cartesian)
        x_cartesian = self.up_cartesian3(x_cartesian, x2_cartesian)
        x_cartesian = self.up_cartesian4(x_cartesian, x1_cartesian)
        logits_cartesian = self.out_cartesian_c(x_cartesian)

        # 极坐标坐标系上采样
        x_polar = self.up_polar1(x5_polar, x4_polar)
        x_polar = self.up_polar2(x_polar, x3_polar)
        x_polar = self.up_polar3(x_polar, x2_polar)
        x_polar = self.up_polar4(x_polar, x1_polar)
        logits_polar = self.out_polar_c(x_polar)
        logits_polar = self.polar_to_cartesian(logits_polar,logits_polar.shape[2],logits_polar.shape[3])

        logits = torch.cat([logits_cartesian, logits_polar], dim=1)  # Shape: [B, 2C, H, W]

        mask = self.merge(logits)


        return mask

    def use_checkpointing(self):
        """启用双路径检查点机制，节省显存"""
        # 笛卡尔路径编码器
        self.inc_cartesian = torch.utils.checkpoint(self.inc_cartesian)
        self.down_cartesian1 = torch.utils.checkpoint(self.down_cartesian1)
        self.down_cartesian2 = torch.utils.checkpoint(self.down_cartesian2)
        self.down_cartesian3 = torch.utils.checkpoint(self.down_cartesian3)
        self.down_cartesian4 = torch.utils.checkpoint(self.down_cartesian4)

        # 笛卡尔路径解码器
        self.up_cartesian1 = torch.utils.checkpoint(self.up_cartesian1)
        self.up_cartesian2 = torch.utils.checkpoint(self.up_cartesian2)
        self.up_cartesian3 = torch.utils.checkpoint(self.up_cartesian3)
        self.up_cartesian4 = torch.utils.checkpoint(self.up_cartesian4)
        self.out_cartesian_c = torch.utils.checkpoint(self.out_cartesian_c)

        # 极坐标路径编码器
        self.inc_polar = torch.utils.checkpoint(self.inc_polar)
        self.down_polar1 = torch.utils.checkpoint(self.down_polar1)
        self.down_polar2 = torch.utils.checkpoint(self.down_polar2)
        self.down_polar3 = torch.utils.checkpoint(self.down_polar3)
        self.down_polar4 = torch.utils.checkpoint(self.down_polar4)

        # 极坐标路径解码器
        self.up_polar1 = torch.utils.checkpoint(self.up_polar1)
        self.up_polar2 = torch.utils.checkpoint(self.up_polar2)
        self.up_polar3 = torch.utils.checkpoint(self.up_polar3)
        self.up_polar4 = torch.utils.checkpoint(self.up_polar4)
        self.out_polar_c = torch.utils.checkpoint(self.out_polar_c)

class EncoderDoubleCoordAttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(EncoderDoubleCoordAttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.cartesian_to_polar = cartesian_to_polar#  笛卡尔系转极坐标系
        self.polar_to_cartesian = polar_to_cartesian # 极坐标系转笛卡尔坐标系


        self.inc_cartesian = (DoubleConv(n_channels, 64))
        self.down_cartesian1 = (Down(64, 128))
        self.down_cartesian2 = (Down(128, 256))
        self.down_cartesian3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_cartesian4 = (Down(512, 1024 // factor))
        self.up_cartesian1 = (Up(1024, 512 // factor, bilinear))
        self.up_cartesian2 = (Up(512, 256 // factor, bilinear))
        self.up_cartesian3 = (Up(256, 128 // factor, bilinear))
        self.up_cartesian4 = (Up(128, 64, bilinear))
        self.out_cartesian_c = (OutConv(64, n_classes))

        self.inc_polar = (DoubleConv(n_channels, 64))
        self.down_polar1 = (Down(64, 128))
        self.down_polar2 = (Down(128, 256))
        self.down_polar3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_polar4 = (Down(512, 1024 // factor))
        self.up_polar1 = (Up(1024, 512 // factor, bilinear))
        self.up_polar2 = (Up(512, 256 // factor, bilinear))
        self.up_polar3 = (Up(256, 128 // factor, bilinear))
        self.up_polar4 = (Up(128, 64, bilinear))
        self.out_polar_c = (OutConv(64, n_classes))

        #双特征图注意力
        self.coord_att1 = DoubleCoordAtt(inp=64, oup=64)
        self.coord_att2 = DoubleCoordAtt(inp=128, oup=128)
        self.coord_att3 = DoubleCoordAtt(inp=256, oup=256)
        self.coord_att4 = DoubleCoordAtt(inp=512, oup=512)
        self.coord_att5 = DoubleCoordAtt(inp=1024, oup=1024)

        self.merge = nn.Conv2d(2*n_classes, n_classes, kernel_size=3, padding=1) #合并两个特征图


    def forward(self, x):
        x_cartesian = x
        x_polar = x

        #笛卡尔坐标系下采样 和 极坐标坐标系下采样
        x1_cartesian = self.inc_cartesian(x_cartesian)
        x_polar = self.cartesian_to_polar(x_polar)
        x1_polar = self.inc_polar(x_polar)
        t_cartesian, t_polar = self.coord_att1(x1_cartesian, x1_polar)
        x1_cartesian, x1_polar = t_cartesian+x1_cartesian, t_polar+x1_polar

        x2_cartesian = self.down_cartesian1(x1_cartesian)
        x2_polar = self.down_polar1(x1_polar)
        t_cartesian, t_polar = self.coord_att2(x2_cartesian, x2_polar)
        x2_cartesian, x2_polar = t_cartesian + x2_cartesian, t_polar + x2_polar


        x3_cartesian = self.down_cartesian2(x2_cartesian)
        x3_polar = self.down_polar2(x2_polar)
        t_cartesian, t_polar = self.coord_att3(x3_cartesian, x3_polar)
        x3_cartesian, x3_polar = t_cartesian + x3_cartesian, t_polar + x3_polar

        x4_cartesian = self.down_cartesian3(x3_cartesian)
        x4_polar = self.down_polar3(x3_polar)
        t_cartesian, t_polar = self.coord_att4(x4_cartesian, x4_polar)
        x4_cartesian, x4_polar = t_cartesian + x4_cartesian, t_polar + x4_polar

        x5_cartesian = self.down_cartesian4(x4_cartesian)
        x5_polar = self.down_polar4(x4_polar)
        t_cartesian, t_polar = self.coord_att5(x5_cartesian, x5_polar)
        x5_cartesian, x5_polar = t_cartesian + x5_cartesian, t_polar + x5_polar

        # 笛卡尔坐标系上采样
        x_cartesian = self.up_cartesian1(x5_cartesian, x4_cartesian)
        x_cartesian = self.up_cartesian2(x_cartesian, x3_cartesian)
        x_cartesian = self.up_cartesian3(x_cartesian, x2_cartesian)
        x_cartesian = self.up_cartesian4(x_cartesian, x1_cartesian)
        logits_cartesian = self.out_cartesian_c(x_cartesian)

        # 极坐标坐标系上采样
        x_polar = self.up_polar1(x5_polar, x4_polar)
        x_polar = self.up_polar2(x_polar, x3_polar)
        x_polar = self.up_polar3(x_polar, x2_polar)
        x_polar = self.up_polar4(x_polar, x1_polar)
        logits_polar = self.out_polar_c(x_polar)
        logits_polar = self.polar_to_cartesian(logits_polar,logits_polar.shape[2],logits_polar.shape[3])

        logits = torch.cat([logits_cartesian, logits_polar], dim=1)  # Shape: [B, 2C, H, W]

        mask = self.merge(logits)


        return mask

    def use_checkpointing(self):
        """启用双路径检查点机制，节省显存"""
        # 笛卡尔路径编码器
        self.inc_cartesian = torch.utils.checkpoint(self.inc_cartesian)
        self.down_cartesian1 = torch.utils.checkpoint(self.down_cartesian1)
        self.down_cartesian2 = torch.utils.checkpoint(self.down_cartesian2)
        self.down_cartesian3 = torch.utils.checkpoint(self.down_cartesian3)
        self.down_cartesian4 = torch.utils.checkpoint(self.down_cartesian4)

        # 笛卡尔路径解码器
        self.up_cartesian1 = torch.utils.checkpoint(self.up_cartesian1)
        self.up_cartesian2 = torch.utils.checkpoint(self.up_cartesian2)
        self.up_cartesian3 = torch.utils.checkpoint(self.up_cartesian3)
        self.up_cartesian4 = torch.utils.checkpoint(self.up_cartesian4)
        self.out_cartesian_c = torch.utils.checkpoint(self.out_cartesian_c)

        # 极坐标路径编码器
        self.inc_polar = torch.utils.checkpoint(self.inc_polar)
        self.down_polar1 = torch.utils.checkpoint(self.down_polar1)
        self.down_polar2 = torch.utils.checkpoint(self.down_polar2)
        self.down_polar3 = torch.utils.checkpoint(self.down_polar3)
        self.down_polar4 = torch.utils.checkpoint(self.down_polar4)

        # 极坐标路径解码器
        self.up_polar1 = torch.utils.checkpoint(self.up_polar1)
        self.up_polar2 = torch.utils.checkpoint(self.up_polar2)
        self.up_polar3 = torch.utils.checkpoint(self.up_polar3)
        self.up_polar4 = torch.utils.checkpoint(self.up_polar4)
        self.out_polar_c = torch.utils.checkpoint(self.out_polar_c)

        self.coord_att1 = torch.utils.checkpoint(self.coord_att1)
        self.coord_att2 = torch.utils.checkpoint(self.coord_att2)
        self.coord_att3 = torch.utils.checkpoint(self.coord_att3)
        self.coord_att4 = torch.utils.checkpoint(self.coord_att4)
        self.coord_att5 = torch.utils.checkpoint(self.coord_att5)

class FullDoubleCoordAttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(FullDoubleCoordAttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.cartesian_to_polar = cartesian_to_polar#  笛卡尔系转极坐标系
        self.polar_to_cartesian = polar_to_cartesian # 极坐标系转笛卡尔坐标系


        self.inc_cartesian = (DoubleConv(n_channels, 64))
        self.down_cartesian1 = (Down(64, 128))
        self.down_cartesian2 = (Down(128, 256))
        self.down_cartesian3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_cartesian4 = (Down(512, 1024 // factor))
        self.up_cartesian1 = (Up(1024, 512 // factor, bilinear))
        self.up_cartesian2 = (Up(512, 256 // factor, bilinear))
        self.up_cartesian3 = (Up(256, 128 // factor, bilinear))
        self.up_cartesian4 = (Up(128, 64, bilinear))
        self.out_cartesian_c = (OutConv(64, n_classes))

        self.inc_polar = (DoubleConv(n_channels, 64))
        self.down_polar1 = (Down(64, 128))
        self.down_polar2 = (Down(128, 256))
        self.down_polar3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down_polar4 = (Down(512, 1024 // factor))
        self.up_polar1 = (Up(1024, 512 // factor, bilinear))
        self.up_polar2 = (Up(512, 256 // factor, bilinear))
        self.up_polar3 = (Up(256, 128 // factor, bilinear))
        self.up_polar4 = (Up(128, 64, bilinear))
        self.out_polar_c = (OutConv(64, n_classes))

        #双特征图注意力
        self.coord_att1 = DoubleCoordAtt(inp=64, oup=64)
        self.coord_att2 = DoubleCoordAtt(inp=128, oup=128)
        self.coord_att3 = DoubleCoordAtt(inp=256, oup=256)
        self.coord_att4 = DoubleCoordAtt(inp=512, oup=512)
        self.coord_att5 = DoubleCoordAtt(inp=1024, oup=1024)

        # Decoder部分的双坐标注意力模块
        self.coord_att_up1 = DoubleCoordAtt(inp=512 // factor, oup=512 // factor)  # 对应up1层
        self.coord_att_up2 = DoubleCoordAtt(inp=256 // factor, oup=256 // factor)  # 对应up2层
        self.coord_att_up3 = DoubleCoordAtt(inp=128 // factor, oup=128 // factor)  # 对应up3层
        self.coord_att_up4 = DoubleCoordAtt(inp=64, oup=64)  # 对应up4层

        self.merge = nn.Conv2d(2*n_classes, n_classes, 3, padding=1)

    def forward(self, x):
        x_cartesian = x
        x_polar = x

        #笛卡尔坐标系下采样 和 极坐标坐标系下采样
        x1_cartesian = self.inc_cartesian(x_cartesian)
        x_polar = self.cartesian_to_polar(x_polar)
        x1_polar = self.inc_polar(x_polar)
        t_cartesian, t_polar = self.coord_att1(x1_cartesian, x1_polar)
        x1_cartesian, x1_polar = t_cartesian+x1_cartesian, t_polar+x1_polar

        x2_cartesian = self.down_cartesian1(x1_cartesian)
        x2_polar = self.down_polar1(x1_polar)
        t_cartesian, t_polar = self.coord_att2(x2_cartesian, x2_polar)
        x2_cartesian, x2_polar = t_cartesian + x2_cartesian, t_polar + x2_polar


        x3_cartesian = self.down_cartesian2(x2_cartesian)
        x3_polar = self.down_polar2(x2_polar)
        t_cartesian, t_polar = self.coord_att3(x3_cartesian, x3_polar)
        x3_cartesian, x3_polar = t_cartesian + x3_cartesian, t_polar + x3_polar

        x4_cartesian = self.down_cartesian3(x3_cartesian)
        x4_polar = self.down_polar3(x3_polar)
        t_cartesian, t_polar = self.coord_att4(x4_cartesian, x4_polar)
        x4_cartesian, x4_polar = t_cartesian + x4_cartesian, t_polar + x4_polar

        x5_cartesian = self.down_cartesian4(x4_cartesian)
        x5_polar = self.down_polar4(x4_polar)
        t_cartesian, t_polar = self.coord_att5(x5_cartesian, x5_polar)
        x5_cartesian, x5_polar = t_cartesian + x5_cartesian, t_polar + x5_polar

        # ====================== 修正后的Decoder部分 ======================
        # 初始化解码器特征
        d_cartesian = x5_cartesian  # 编码器最后一层输出 [1024, H/16, W/16]
        d_polar = x5_polar

        # 定义各层的跳跃连接引用
        skip_connections_cartesian = [x4_cartesian, x3_cartesian, x2_cartesian, x1_cartesian]
        skip_connections_polar = [x4_polar, x3_polar, x2_polar, x1_polar]

        # 按层次处理解码过程（从深层到浅层）
        for layer_idx in range(4):
            # 上采样操作 ------------------------------------------------
            up_layer = getattr(self, f'up_cartesian{layer_idx + 1}')
            skip_cart = skip_connections_cartesian[layer_idx]  # 倒序取跳跃连接
            d_cartesian = up_layer(d_cartesian, skip_cart)

            up_polar_layer = getattr(self, f'up_polar{layer_idx + 1}')
            skip_polar = skip_connections_polar[layer_idx]
            d_polar = up_polar_layer(d_polar, skip_polar)

            # 注意力交互 ------------------------------------------------
            # 保存原始特征用于残差连接
            original_cart = d_cartesian.clone().detach()
            original_polar = d_polar.clone().detach()

            # 应用双路径注意力
            att_layer = getattr(self, f'coord_att_up{layer_idx + 1}')
            att_cart, att_polar = att_layer(d_cartesian, d_polar)

            # 残差连接（注意力输出 + 原始上采样输出）
            d_cartesian = att_cart + original_cart
            d_polar = att_polar + original_polar

        # ==================== 最终输出部分 ====================
        # 笛卡尔路径最终输出
        logits_cartesian = self.out_cartesian_c(d_cartesian)

        # 极坐标路径处理
        logits_polar = self.out_polar_c(d_polar)
        logits_polar = self.polar_to_cartesian(
            logits_polar,
            logits_cartesian.shape[2],  # 保持与笛卡尔输出尺寸一致
            logits_cartesian.shape[3]
        )

        # 特征融合
        combined = torch.cat([logits_cartesian, logits_polar], dim=1)
        final_mask = self.merge(combined)

        return final_mask

    def use_checkpointing(self):
        """启用双路径检查点机制，节省显存"""
        # 笛卡尔路径编码器
        self.inc_cartesian = torch.utils.checkpoint(self.inc_cartesian)
        self.down_cartesian1 = torch.utils.checkpoint(self.down_cartesian1)
        self.down_cartesian2 = torch.utils.checkpoint(self.down_cartesian2)
        self.down_cartesian3 = torch.utils.checkpoint(self.down_cartesian3)
        self.down_cartesian4 = torch.utils.checkpoint(self.down_cartesian4)

        # 笛卡尔路径解码器
        self.up_cartesian1 = torch.utils.checkpoint(self.up_cartesian1)
        self.up_cartesian2 = torch.utils.checkpoint(self.up_cartesian2)
        self.up_cartesian3 = torch.utils.checkpoint(self.up_cartesian3)
        self.up_cartesian4 = torch.utils.checkpoint(self.up_cartesian4)
        self.out_cartesian_c = torch.utils.checkpoint(self.out_cartesian_c)

        # 极坐标路径编码器
        self.inc_polar = torch.utils.checkpoint(self.inc_polar)
        self.down_polar1 = torch.utils.checkpoint(self.down_polar1)
        self.down_polar2 = torch.utils.checkpoint(self.down_polar2)
        self.down_polar3 = torch.utils.checkpoint(self.down_polar3)
        self.down_polar4 = torch.utils.checkpoint(self.down_polar4)

        # 极坐标路径解码器
        self.up_polar1 = torch.utils.checkpoint(self.up_polar1)
        self.up_polar2 = torch.utils.checkpoint(self.up_polar2)
        self.up_polar3 = torch.utils.checkpoint(self.up_polar3)
        self.up_polar4 = torch.utils.checkpoint(self.up_polar4)
        self.out_polar_c = torch.utils.checkpoint(self.out_polar_c)

        self.coord_att1 = torch.utils.checkpoint(self.coord_att1)
        self.coord_att2 = torch.utils.checkpoint(self.coord_att2)
        self.coord_att3 = torch.utils.checkpoint(self.coord_att3)
        self.coord_att4 = torch.utils.checkpoint(self.coord_att4)
        self.coord_att5 = torch.utils.checkpoint(self.coord_att5)

        self.coord_att_up1 = torch.utils.checkpoint(self.coord_att_up1)
        self.coord_att_up2 = torch.utils.checkpoint(self.coord_att_up2)
        self.coord_att_up3 = torch.utils.checkpoint(self.coord_att_up3)
        self.coord_att_up4 = torch.utils.checkpoint(self.coord_att_up4)


class DecoderDoubleCoordAttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DecoderDoubleCoordAttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 坐标系转换模块保持不变
        self.cartesian_to_polar = cartesian_to_polar
        self.polar_to_cartesian = polar_to_cartesian

        # ------------------- 移除了Encoder注意力相关定义 -------------------
        # Cartesian Encoder
        self.inc_cartesian = DoubleConv(n_channels, 64)
        self.down_cartesian1 = Down(64, 128)
        self.down_cartesian2 = Down(128, 256)
        self.down_cartesian3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down_cartesian4 = Down(512, 1024 // factor)

        # Polar Encoder
        self.inc_polar = DoubleConv(n_channels, 64)
        self.down_polar1 = Down(64, 128)
        self.down_polar2 = Down(128, 256)
        self.down_polar3 = Down(256, 512)
        self.down_polar4 = Down(512, 1024 // factor)

        # ------------------- Decoder部分保持不变 -------------------
        # Cartesian Decoder
        self.up_cartesian1 = Up(1024, 512 // factor, bilinear)
        self.up_cartesian2 = Up(512, 256 // factor, bilinear)
        self.up_cartesian3 = Up(256, 128 // factor, bilinear)
        self.up_cartesian4 = Up(128, 64, bilinear)
        self.out_cartesian_c = OutConv(64, n_classes)

        # Polar Decoder
        self.up_polar1 = Up(1024, 512 // factor, bilinear)
        self.up_polar2 = Up(512, 256 // factor, bilinear)
        self.up_polar3 = Up(256, 128 // factor, bilinear)
        self.up_polar4 = Up(128, 64, bilinear)
        self.out_polar_c = OutConv(64, n_classes)

        # ------------------- 仅保留Decoder注意力 -------------------
        # Decoder双路径注意力模块
        self.coord_att_up1 = DoubleCoordAtt(inp=512 // factor, oup=512 // factor)
        self.coord_att_up2 = DoubleCoordAtt(inp=256 // factor, oup=256 // factor)
        self.coord_att_up3 = DoubleCoordAtt(inp=128 // factor, oup=128 // factor)
        self.coord_att_up4 = DoubleCoordAtt(inp=64, oup=64)

        # 特征融合层保持不变
        self.merge = nn.Conv2d(2*n_classes, n_classes, 3, padding=1)

    def forward(self, x):
        x_cartesian = x
        x_polar = x

        # ------------------- 移除了Encoder的注意力交互 -------------------
        # Cartesian Encoder
        x1_cartesian = self.inc_cartesian(x_cartesian)
        x_polar_trans = self.cartesian_to_polar(x_polar)
        x1_polar = self.inc_polar(x_polar_trans)

        x2_cartesian = self.down_cartesian1(x1_cartesian)
        x2_polar = self.down_polar1(x1_polar)

        x3_cartesian = self.down_cartesian2(x2_cartesian)
        x3_polar = self.down_polar2(x2_polar)

        x4_cartesian = self.down_cartesian3(x3_cartesian)
        x4_polar = self.down_polar3(x3_polar)

        x5_cartesian = self.down_cartesian4(x4_cartesian)
        x5_polar = self.down_polar4(x4_polar)

        # ------------------- Decoder部分保持不变 -------------------
        d_cartesian = x5_cartesian
        d_polar = x5_polar

        skip_connections_cartesian = [x4_cartesian, x3_cartesian, x2_cartesian, x1_cartesian]
        skip_connections_polar = [x4_polar, x3_polar, x2_polar, x1_polar]

        for layer_idx in range(4):
            # 上采样操作
            up_cart = getattr(self, f'up_cartesian{layer_idx+1}')
            d_cartesian = up_cart(d_cartesian, skip_connections_cartesian[layer_idx])

            up_polar = getattr(self, f'up_polar{layer_idx+1}')
            d_polar = up_polar(d_polar, skip_connections_polar[layer_idx])

            # 注意力交互（保留）
            original_cart = d_cartesian.clone().detach()
            original_polar = d_polar.clone().detach()
            att_layer = getattr(self, f'coord_att_up{layer_idx+1}')
            att_cart, att_polar = att_layer(d_cartesian, d_polar)
            d_cartesian = att_cart + original_cart
            d_polar = att_polar + original_polar

        # 最终输出
        logits_cartesian = self.out_cartesian_c(d_cartesian)
        logits_polar = self.out_polar_c(d_polar)
        logits_polar = self.polar_to_cartesian(
            logits_polar,
            logits_cartesian.shape[2],
            logits_cartesian.shape[3]
        )
        return self.merge(torch.cat([logits_cartesian, logits_polar], dim=1))