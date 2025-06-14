import argparse
import json
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.data_loading import BasicDataset
from utils.testScore import test_single_volume
from model.swin_unet import *
from unet.unet_model import *

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data', help='测试数据根目录（包含imgs/label_1/label_2）')
parser.add_argument('--dataset', type=str,
                    default='Own', help='数据集名称')
parser.add_argument('--num_classes', type=int,
                    default=2, help='标签类别数（以第一个标签为准）')
parser.add_argument('--list_dir', type=str,
                    default='', help='（未使用可忽略）')
parser.add_argument('--max_epochs', type=int, default=50, help='最大训练轮数（仅用于路径生成）')
parser.add_argument('--batch_size', type=int, default=24, help='批大小')
parser.add_argument('--img_size', type=int, default=256, help='输入图像尺寸')
parser.add_argument('--is_savenii', action="store_true", help='是否保存预测结果')
parser.add_argument('--n_skip', type=int, default=3, help='跳跃连接数')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT模型名称')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='预测结果保存路径')
parser.add_argument('--deterministic', type=int, default=1, help='确定性训练')
parser.add_argument('--base_lr', type=float, default=0.01, help='基础学习率')
parser.add_argument('--seed', type=int, default=0, help='随机种子')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT分块大小')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    """评估模型性能并记录指标

    Args:
        args: 命令行参数
        model: 加载的模型
        test_save_path: 预测结果保存路径

    Returns:
        dict: 包含各类别和整体指标的字典
    """
    # 初始化日志记录
    logging.info("\n" + "=" * 50)
    logging.info(f"开始评估模型: {args.exp}")
    logging.info(f"数据路径: {args.volume_path}")
    logging.info(f"类别数: {args.num_classes}")

    # 加载数据集
    try:
        db_full = BasicDataset(
            os.path.join(args.volume_path, "val_imgs"),
            os.path.join(args.volume_path, "val_masks"))
    except Exception as e:
        logging.error(f"数据加载失败: {str(e)}")
        raise


    val_set = db_full

    # 创建数据加载器
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化指标存储
    all_metrics = {
        'dice': [[] for _ in range(args.num_classes - 1)],
        'hd95': [[] for _ in range(args.num_classes - 1)],
        'precision': [[] for _ in range(args.num_classes - 1)],
        'sensitivity': [[] for _ in range(args.num_classes - 1)],
        'iou': [[] for _ in range(args.num_classes - 1)],
        'volume_similarity': [[] for _ in range(args.num_classes - 1)]
    }
    case_results = []

    # 开始评估
    model.eval()
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="评估中"):
            image, label = batch["image"], batch["mask"]
            image, label = image.cuda(non_blocking=True), label.cuda(non_blocking=True)

            case_name = f"case_{i_batch}"

            # 获取预测结果和指标
            metrics = test_single_volume(
                image, label, model,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
                test_save_path=test_save_path,
                case=case_name
            )

            # 记录每个样本的结果
            case_result = {'name': case_name}
            for cls_idx in range(args.num_classes - 1):
                dice, hd95, precision, sensitivity, iou, vs = metrics[cls_idx]

                # 存储到all_metrics
                all_metrics['dice'][cls_idx].append(dice)
                all_metrics['hd95'][cls_idx].append(hd95)
                all_metrics['precision'][cls_idx].append(precision)
                all_metrics['sensitivity'][cls_idx].append(sensitivity)
                all_metrics['iou'][cls_idx].append(iou)
                all_metrics['volume_similarity'][cls_idx].append(vs)

                # 更新case_result
                case_result[f'cls_{cls_idx + 1}_dice'] = dice
                case_result[f'cls_{cls_idx + 1}_hd95'] = hd95
                case_result[f'cls_{cls_idx + 1}_precision'] = precision
                case_result[f'cls_{cls_idx + 1}_sensitivity'] = sensitivity
                case_result[f'cls_{cls_idx + 1}_iou'] = iou
                case_result[f'cls_{cls_idx + 1}_vs'] = vs

            case_results.append(case_result)
            logging.info(
                f"{case_name} - " +
                " | ".join([
                    f"类别{cls_idx + 1}: "
                    f"Dice={metrics[cls_idx][0]:.4f}, "
                    f"HD95={metrics[cls_idx][1]:.4f}, "
                    f"Precision={metrics[cls_idx][2]:.4f}, "
                    f"Sensitivity={metrics[cls_idx][3]:.4f}, "
                    f"IoU={metrics[cls_idx][4]:.4f}, "
                    f"VS={metrics[cls_idx][5]:.4f}"
                    for cls_idx in range(args.num_classes - 1)
                ])
            )

    # 计算统计指标
    results = {
        'per_class': {},
        'overall': {
            'mean_dice': 0,
            'mean_hd95': 0,
            'mean_precision': 0,
            'mean_sensitivity': 0,
            'mean_iou': 0,
            'mean_vs': 0
        },
        'case_results': case_results
    }

    # 计算每个类别的平均指标
    for cls_idx in range(args.num_classes - 1):
        # 计算均值
        cls_dice = np.mean(all_metrics['dice'][cls_idx])
        cls_hd95 = np.mean(all_metrics['hd95'][cls_idx])
        cls_precision = np.mean(all_metrics['precision'][cls_idx])
        cls_sensitivity = np.mean(all_metrics['sensitivity'][cls_idx])
        cls_iou = np.mean(all_metrics['iou'][cls_idx])
        cls_vs = np.mean(all_metrics['volume_similarity'][cls_idx])

        # 计算标准差
        std_dice = np.std(all_metrics['dice'][cls_idx])
        std_hd95 = np.std(all_metrics['hd95'][cls_idx])
        std_precision = np.std(all_metrics['precision'][cls_idx])
        std_sensitivity = np.std(all_metrics['sensitivity'][cls_idx])
        std_iou = np.std(all_metrics['iou'][cls_idx])
        std_vs = np.std(all_metrics['volume_similarity'][cls_idx])

        # 存储到per_class
        results['per_class'][f'cls_{cls_idx + 1}'] = {
            'mean_dice': float(cls_dice),
            'std_dice': float(std_dice),
            'mean_hd95': float(cls_hd95),
            'std_hd95': float(std_hd95),
            'mean_precision': float(cls_precision),
            'std_precision': float(std_precision),
            'mean_sensitivity': float(cls_sensitivity),
            'std_sensitivity': float(std_sensitivity),
            'mean_iou': float(cls_iou),
            'std_iou': float(std_iou),
            'mean_vs': float(cls_vs),
            'std_vs': float(std_vs)
        }

        # 累加整体指标
        results['overall']['mean_dice'] += cls_dice
        results['overall']['mean_hd95'] += cls_hd95
        results['overall']['mean_precision'] += cls_precision
        results['overall']['mean_sensitivity'] += cls_sensitivity
        results['overall']['mean_iou'] += cls_iou
        results['overall']['mean_vs'] += cls_vs

    # 计算整体均值（按类别数平均）
    num_classes = args.num_classes - 1
    results['overall']['mean_dice'] /= num_classes
    results['overall']['mean_hd95'] /= num_classes
    results['overall']['mean_precision'] /= num_classes
    results['overall']['mean_sensitivity'] /= num_classes
    results['overall']['mean_iou'] /= num_classes
    results['overall']['mean_vs'] /= num_classes

    # 记录汇总结果
    logging.info("\n评估结果汇总:")
    for cls_idx in range(args.num_classes - 1):
        cls_info = results['per_class'][f'cls_{cls_idx + 1}']
        logging.info(
            f"类别 {cls_idx + 1} - "
            f"Dice: {cls_info['mean_dice']:.4f}±{cls_info['std_dice']:.4f} | "
            f"HD95: {cls_info['mean_hd95']:.4f}±{cls_info['std_hd95']:.4f} | "
            f"Precision: {cls_info['mean_precision']:.4f}±{cls_info['std_precision']:.4f} | "
            f"Sensitivity: {cls_info['mean_sensitivity']:.4f}±{cls_info['std_sensitivity']:.4f} | "
            f"IoU: {cls_info['mean_iou']:.4f}±{cls_info['std_iou']:.4f} | "
            f"VS: {cls_info['mean_vs']:.4f}±{cls_info['std_vs']:.4f}"
        )

    logging.info(
        f"整体指标 - "
        f"Dice: {results['overall']['mean_dice']:.4f} | "
        f"HD95: {results['overall']['mean_hd95']:.4f} | "
        f"Precision: {results['overall']['mean_precision']:.4f} | "
        f"Sensitivity: {results['overall']['mean_sensitivity']:.4f} | "
        f"IoU: {results['overall']['mean_iou']:.4f} | "
        f"VS: {results['overall']['mean_vs']:.4f}"
    )
    logging.info("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    # 确定性设置（与训练一致）
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置（关键：适配DoubleDataset）
    dataset_config = {
        'Own': {
            'Dataset': BasicDataset,  # 修改为DoubleDataset
            'volume_path': './data',
            'num_classes': 2,  # 以第一个标签的类别数为准
            'z_spacing': 1,
        }
    }

    # 应用配置
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    # 模型路径（根据训练生成的路径调整）
    args.exp = 'DoubleUNet'
    snapshot_path = "./checkpoints2/checkpoint_epoch100.pth"

    # 加载模型
    input_channels = 3
    #net = DecoderDoubleCoordAttUNet(n_channels=input_channels, n_classes=args.num_classes, bilinear=False).cuda()

    # 创建模型实例
    net = DoubleUNet(n_channels=3, n_classes=args.num_classes, bilinear=False).cuda()

    #print("模型第一层权重形状:", net.inc.double_conv[0].weight.shape)
    snapshot = torch.load(snapshot_path)
    checkpoint = torch.load(snapshot_path, map_location='cuda')

    if 'mask_values' in snapshot:
        snapshot.pop('mask_values')  # 移除无关键值
    model_state_dict = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = net.load_state_dict(
        model_state_dict,
        strict=True  # 严格匹配模式
    )

    # 调试信息输出
    print(f"Missing keys: {missing_keys}")  # 应为空列表
    print(f"Unexpected keys: {unexpected_keys}")  # 应为空列表

    # 日志配置
    log_folder = f'./test_log/test_log_{args.exp}'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, 'test_log.txt'),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    # 预测保存路径
    test_save_path = os.path.join(args.test_save_dir, args.exp)
    if test_save_path:
        os.makedirs(test_save_path, exist_ok=True)

    inference(args, net, test_save_path)
#python testTool2.py
