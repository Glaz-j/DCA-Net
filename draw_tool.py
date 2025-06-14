import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_cases_grid(case_list: list,
                    model_list: list,
                    root_dir: str = "./predictions",
                    img_folder: str = "imgs",
                    mask_folder: str = "masks",
                    pred_suffix: str = "_pred.png",
                    figsize_per: tuple = (3, 3)):
    """
    把多个 case 按行、多个模型按列绘制到同一个图中：
      列顺序：Image | model1_pred | ... | modelN_pred | Mask
      行顺序：case_list 中的先后顺序

    Args:
        case_list:   样本名列表，不带后缀，比如 ["case_1","case_42","case_217"]
        model_list:  模型名列表，决定中间列的顺序
        root_dir:    predictions 根目录
        img_folder:  原图子目录名
        mask_folder: mask 子目录名
        pred_suffix: 预测图后缀，如 "_pred.png"
        figsize_per: 每个子图的 (宽,高)，英寸
    """
    n_rows = len(case_list)
    n_cols = 1 + len(model_list) + 1  # Image + N models + Mask
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per[0]*n_cols,
                                      figsize_per[1]*n_rows),
                             squeeze=False)

    for r, case_id in enumerate(case_list):
        # 1) 原图
        path_img = os.path.join(root_dir, img_folder, case_id + ".png")
        img = np.array(Image.open(path_img))
        ax = axes[r, 0]
        ax.imshow(img)
        ax.set_title("Image" if r==0 else "")
        ax.axis("off")

        # 2) 各模型预测
        for c, model_name in enumerate(model_list, start=1):
            p = os.path.join(root_dir, model_name, case_id + pred_suffix)
            pred = np.array(Image.open(p))
            ax = axes[r, c]
            ax.imshow(pred, cmap="gray")
            ax.set_title(model_name if r==0 else "")
            ax.axis("off")

        # 3) 真值 Mask
        path_mask = os.path.join(root_dir, mask_folder, case_id + ".png")
        m_img = np.array(Image.open(path_mask))
        ax = axes[r, -1]
        ax.imshow(m_img, cmap="gray")
        ax.set_title("Mask" if r==0 else "")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


cases  = ["case_10", "case_22", "case_75","case_122","case_162","case_192","case_217"]
models = ["UNet","UnetPlusPlus","swin-unet","DoubleUNet","EncoderDoubleCoordAttUNet"]
plot_cases_grid(cases, models)
