from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


_BATCH_NORM = nn.BatchNorm2d
_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5,
                                          momentum=0.999))  # PyTorch的默认momentum是0.1, TensorFlow是0.99. DeepLab通常用接近TF的设置

        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))  # 使用 inplace=True 节省内存


# 选项 A: 在 _ImagePool 中移除BN或替换为其他Norm
class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 示例：使用 Conv + ReLU，不使用BN。注意Conv2d的bias需要设为True
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, bias=True), # bias=True if no BN
            nn.ReLU(inplace=True)
        )
        # 或者使用 GroupNorm
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, bias=False),
        #     nn.GroupNorm(num_groups=32, num_channels=out_ch), # num_groups 是一个超参数
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):  # out_ch 通常是 256
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        # 1x1 conv
        self.stages.add_module("c0", _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        # Atrous convolutions
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        # Image pooling
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        # 将所有ASPP分支的输出在通道维度上拼接
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class DeepLabV3(nn.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        # Stride and dilation configuration based on output_stride
        if output_stride == 8:
            s = [1, 2, 1, 1]  # strides for layer2, layer3, layer4, layer5
            d = [1, 1, 2, 4]  # dilations for layer2, layer3, layer4, layer5
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]
        else:
            raise ValueError("output_stride must be 8 or 16.")

        ch = [64 * 2 ** p for p in range(6)]  # [64, 128, 256, 512, 1024, 2048]

        # Backbone (ResNet-like)

        self.add_module("layer1", _Stem(ch[0]))  # Output: H/4, W/4
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))  # ResNet block1, output ch[2]=256
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))  # ResNet block2, output ch[3]=512
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))  # ResNet block3, output ch[4]=1024
        self.add_module(  # ResNet block4, output ch[5]=2048
            "layer5", _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        )

        # ASPP module
        aspp_out_channels = 256  # Standard output channels for each ASPP branch
        self.add_module("aspp", _ASPP(ch[5], aspp_out_channels, atrous_rates))

        # Output head
        # Number of branches in ASPP: 1 (1x1 conv) + len(atrous_rates) (atrous convs) + 1 (image pooling)
        aspp_concat_ch = aspp_out_channels * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(aspp_concat_ch, aspp_out_channels, 1, 1, 0, 1))  # Project ASPP output
        self.add_module("fc2", nn.Conv2d(aspp_out_channels, n_classes, kernel_size=1))  # Final classification layer

        # Initialize weights (optional, but good practice)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (_BATCH_NORM, nn.GroupNorm)):  # nn.GroupNorm can be an alternative to SyncBatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        original_size = x.shape[2:]  # H, W of the input image

        # Pass input through all modules defined in __init__ (stem, reslayers, aspp, fc1, fc2)
        # This is the default behavior of nn.Sequential.forward()
        # The output 'out' will have dimensions [N, n_classes, H_feat, W_feat]
        # where H_feat = H / output_stride, W_feat = W / output_stride
        out = super().forward(x)

        # Upsample to original image size
        # Using align_corners=True to match DeepLabv2's behavior if that's intended.
        # Modern implementations often use align_corners=False.
        out = F.interpolate(
            out,
            size=original_size,
            mode='bilinear',
            align_corners=True  # Consistent with your DeepLabV2 snippet
        )
        return out


class _Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)  # stride affects spatial dim here
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)  # dilation here
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)  # No ReLU before shortcut
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)  # stride also here for projection
            if downsample  # downsample is True if in_ch != out_ch or stride != 1
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)  # ReLU after addition


class _ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids), "n_layers and len(multi_grids) must be equal."

        # Downsampling (stride > 1) and initial dilation apply only to the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                _Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),  # input channels change after first block
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),  # stride only for the first block of the layer
                    dilation=dilation * multi_grids[i],
                    # downsample is True if input channels or spatial dimensions change for the shortcut
                    downsample=(True if i == 0 and (in_ch != out_ch or stride != 1) else False) if i == 0 else False,

                ),
            )
        # A more robust check for downsample in the first block:
        # The first block needs a projection shortcut if its input channels are different from output channels,
        # or if its stride is > 1.
        first_block_downsample = (stride != 1) or (in_ch != out_ch)

        # Re-define the first block with corrected downsample logic if necessary
        # (The original code was mostly correct, but let's be explicit)
        self._modules["block1"] = _Bottleneck(
            in_ch=in_ch,
            out_ch=out_ch,
            stride=stride,
            dilation=dilation * multi_grids[0],
            downsample=first_block_downsample
        )

        # For subsequent blocks, downsample is False (no change in channels or stride within the layer after the first block)
        # and in_ch becomes out_ch.
        current_in_ch = out_ch
        for i in range(1, n_layers):
            self._modules["block{}".format(i + 1)] = _Bottleneck(
                in_ch=current_in_ch,
                out_ch=out_ch,
                stride=1,
                dilation=dilation * multi_grids[i],
                downsample=False
            )


class _Stem(nn.Sequential):
    """
    The 1st conv layer (stem) for DeepLab.
    """

    def __init__(self, out_ch):  # out_ch is typically 64
        super(_Stem, self).__init__()
        # Standard ResNet stem: Conv -> BN -> ReLU -> MaxPool
        self.add_module("conv1", _ConvBnReLU(3, out_ch, 7, 2, 3, 1))  # 7x7 conv, stride 2, padding 3
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))  # MaxPool, stride 2


"""
# Example instantiation:
model = DeepLabV3(
    n_classes=21,  # Example: PASCAL VOC
    n_blocks=[3, 4, 23, 3],  # Corresponds to ResNet-101 backbone structure (block counts for layer2 to layer5)
    atrous_rates=[6, 12, 18],  # Standard atrous rates for ASPP
    multi_grids=[1, 2, 4],  # Multi-grid for the last ResLayer (layer5)
    output_stride=16,  # or 8
)
model.eval()  # Set to evaluation mode

# Create a dummy input image
# For output_stride=16, typical input sizes are multiples of 16 + 1, e.g., 513x513
# For output_stride=8, typical input sizes are multiples of 8 + 1
image_size = 513
image = torch.randn(1, 3, image_size, image_size)  # Batch size 1, 3 channels (RGB)

print(f"Input image shape: {image.shape}")

# Get the output from the model
with torch.no_grad():  # Important for evaluation to save memory and computation
    output = model(image)

print(f"Output tensor shape: {output.shape}")  # Should be [1, n_classes, image_size, image_size]

# You can also print the model structure
# print(model)

# Test with output_stride = 8
print("\n--- Testing with output_stride = 8 ---")
model_os8 = DeepLabV3(
    n_classes=21,
    n_blocks=[3, 4, 23, 3],
    atrous_rates=[12, 24, 36],  # Rates are typically doubled for OS=8 compared to OS=16
    multi_grids=[1, 2, 4],
    output_stride=8,
)
model_os8.eval()
image_os8 = torch.randn(1, 3, image_size, image_size)
print(f"Input image shape (OS=8): {image_os8.shape}")
with torch.no_grad():
    output_os8 = model_os8(image_os8)
print(f"Output tensor shape (OS=8): {output_os8.shape}")"""