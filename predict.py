import argparse
import logging
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
from unet import *
from utils.utils import plot_img_and_mask


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        mask = torch.sigmoid(output) > out_threshold if net.n_classes == 1 else output.argmax(dim=1)

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the model file')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Directory to save output masks')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize results')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save outputs')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability threshold for mask')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    for i, v in enumerate(mask_values):
        out[mask == i] = v
    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 准备目录
    os.makedirs(args.output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(args.input_dir, '*.[pj][np]g'))  # 匹配png/jpg/jpeg

    # 初始化模型
    net = UNetCroodAttention(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    # 批量处理
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            mask = predict_img(net, img, device, args.scale, args.mask_threshold)

            if not args.no_save:
                out_path = os.path.join(args.output_dir, os.path.basename(img_path))
                mask_to_image(mask, mask_values).save(out_path)
                logging.info(f'Saved mask to {out_path}')

            if args.viz:
                plot_img_and_mask(img, mask)

        except Exception as e:
            logging.error(f'Error processing {img_path}: {str(e)}')