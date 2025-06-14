import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def cartesian_to_polar(img_tensor):
    input_shape = img_tensor.shape
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    B, C, H, W = img_tensor.shape

    # 将尺寸转换为张量并保留设备信息
    W_tensor = torch.tensor(W - 1.0, dtype=img_tensor.dtype, device=img_tensor.device)
    H_tensor = torch.tensor(H - 1.0, dtype=img_tensor.dtype, device=img_tensor.device)

    cx = W_tensor / 2.0
    cy = H_tensor / 2.0
    R_max = torch.sqrt((W_tensor / 2.0)**2 + (H_tensor / 2.0)**2)

    i_polar = torch.linspace(0, H-1, H, device=img_tensor.device)
    j_polar = torch.linspace(0, W-1, W, device=img_tensor.device)
    i, j = torch.meshgrid(i_polar, j_polar, indexing='ij')

    r = (i / (H - 1)) * R_max  # 使用张量计算
    theta = j * (2 * math.pi) / W

    x = r * torch.cos(theta) + cx
    y = r * torch.sin(theta) + cy

    x_normalized = (x / (W - 1)) * 2 - 1
    y_normalized = (y / (H - 1)) * 2 - 1

    grid = torch.stack([x_normalized, y_normalized], dim=-1)
    grid = grid.unsqueeze(0).expand(B, H, W, 2)

    polar_tensor = F.grid_sample(img_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    if len(input_shape) == 2:
        polar_tensor = polar_tensor.squeeze(0).squeeze(0)
    elif len(input_shape) == 3:
        polar_tensor = polar_tensor.squeeze(0)
    return polar_tensor

#极坐标系转笛卡尔坐标系
def polar_to_cartesian(polar_tensor, target_h, target_w):
    input_shape = polar_tensor.shape
    if polar_tensor.dim() == 2:
        polar_tensor = polar_tensor.unsqueeze(0).unsqueeze(0)
    elif polar_tensor.dim() == 3:
        polar_tensor = polar_tensor.unsqueeze(0)
    B, C, H_polar, W_polar = polar_tensor.shape

    # 将目标尺寸转换为张量
    target_w_tensor = torch.tensor(target_w - 1.0, dtype=torch.float32, device=polar_tensor.device)
    target_h_tensor = torch.tensor(target_h - 1.0, dtype=torch.float32, device=polar_tensor.device)
    R_max_target = torch.sqrt((target_w_tensor / 2.0)**2 + (target_h_tensor / 2.0)**2)

    x_cart = torch.linspace(0, target_w-1, target_w, device=polar_tensor.device)
    y_cart = torch.linspace(0, target_h-1, target_h, device=polar_tensor.device)
    x, y = torch.meshgrid(x_cart, y_cart, indexing='xy')

    dx = x - (target_w_tensor / 2.0)
    dy = y - (target_h_tensor / 2.0)

    r = torch.sqrt(dx**2 + dy**2)
    theta = torch.atan2(dy, dx) % (2 * math.pi)

    i_polar = (r / R_max_target) * (H_polar - 1)
    j_polar = theta * W_polar / (2 * math.pi)

    i_normalized = (i_polar / (H_polar - 1)) * 2 - 1
    j_normalized = (j_polar / (W_polar - 1)) * 2 - 1

    grid = torch.stack([j_normalized, i_normalized], dim=-1)
    grid = grid.unsqueeze(0).expand(B, target_h, target_w, 2)

    cart_tensor = F.grid_sample(polar_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    if len(input_shape) == 2:
        cart_tensor = cart_tensor.squeeze(0).squeeze(0)
    elif len(input_shape) == 3:
        cart_tensor = cart_tensor.squeeze(0)
    return cart_tensor


