import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class DoubleCoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(DoubleCoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.pool_r = nn.AdaptiveAvgPool2d((None, 1))#r方向
        self.pool_theta = nn.AdaptiveAvgPool2d((1, None))#theta方向


        mip = max(8, inp // reduction)

        self.conv_r = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0) #rxy通道压缩
        self.bn_r = nn.BatchNorm2d(mip)
        self.act_r = h_swish()

        self.conv_theta = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0) #theta xy通道压缩
        self.bn_theta = nn.BatchNorm2d(mip)
        self.act_theta = h_swish()



        self.conv_h_out = nn.Conv2d(mip*2, oup, kernel_size=1, stride=1, padding=0)#*2是因为要拼接两种通道注意力
        self.conv_w_out = nn.Conv2d(mip*2, oup, kernel_size=1, stride=1, padding=0)

        self.conv_r_out = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_theta_out = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)


    def forward(self, x1,x2):
        #默认x1为xy图像，x2为r theta图像
        identity1 = x1
        identity2 = x2

        n,c,h,w = x1.size()#默认nc相同,hw与r,theta相同
        n,c,r,theta = x2.size()


        x_h = self.pool_h(x1)
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)
        x_r = self.pool_r(x2)
        x_theta = self.pool_theta(x2).permute(0, 1, 3, 2)

        y1 = torch.cat([x_h, x_w, x_r], dim=2)#xyr通道
        y1 = self.conv_r(y1)
        y1 = self.bn_r(y1)
        y1 = self.act_r(y1)

        y2 = torch.cat([x_h, x_w, x_theta], dim=2)#xy theta通道
        y2 = self.conv_theta(y2)
        y2 = self.bn_theta(y2)
        y2 = self.act_theta(y2)


        x_h1, x_w1, x_r = torch.split(y1, [h, w, r], dim=2)
        x_w1 = x_w1.permute(0, 1, 3, 2)

        x_h2, x_w2, x_theta = torch.split(y2, [h, w, theta], dim=2)
        x_w2 = x_w2.permute(0, 1, 3, 2)
        x_theta = x_theta.permute(0, 1, 3, 2)

        x_h = torch.cat([x_h1,x_h2], dim=1)
        x_w = torch.cat([x_w1,x_w2], dim=1)

        a_h = self.conv_h_out(x_h).sigmoid()
        a_w = self.conv_w_out(x_w).sigmoid()

        a_r = self.conv_r_out(x_r).sigmoid()
        a_theta = self.conv_theta_out(x_theta).sigmoid()

        out1 = identity1 * a_w * a_h
        out2 = identity2 * a_r * a_theta

        return out1,out2
