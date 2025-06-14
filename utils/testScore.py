import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2


# --- Helper for HD95 calculation using NumPy ---
def hausdorff_distance_95_numpy(pred_mask_np, gt_mask_np, voxelspacing=None):
    """
    Compute the 95th percentile of the Hausdorff Distance between two binary masks using NumPy.
    pred_mask_np: numpy array of the predicted mask
    gt_mask_np: numpy array of the ground truth mask
    voxelspacing: list or tuple of voxel spacings for each dimension (e.g., [z,y,x])
                  If None, isotropic spacing of 1 is assumed.

    Returns:
    hd95: float, the 95th percentile Hausdorff distance.
          Returns 0 if either mask is empty, aligning with some typical behaviors
          or a large value might be more appropriate if one is empty and not other.
          Current behavior: if any mask has no points, hd95 is 0 here.
    """
    if not np.any(pred_mask_np) or not np.any(gt_mask_np):
        return 0.0

    coords_pred = np.array(np.where(pred_mask_np)).T
    coords_gt = np.array(np.where(gt_mask_np)).T

    if coords_pred.shape[0] == 0 or coords_gt.shape[0] == 0:
        return 0.0

    if voxelspacing is not None:
        voxelspacing = np.array(voxelspacing)
        if voxelspacing.shape[0] != coords_pred.shape[1]:
            raise ValueError(
                f"Voxelspacing dimension {voxelspacing.shape[0]} does not match coordinate dimension {coords_pred.shape[1]}")
        coords_pred = coords_pred * voxelspacing
        coords_gt = coords_gt * voxelspacing

    # Compute distances from pred to gt
    dists_pred_to_gt = np.zeros(coords_pred.shape[0])
    for i, point_p in enumerate(coords_pred):
        dists_pred_to_gt[i] = np.min(np.sqrt(np.sum((coords_gt - point_p) ** 2, axis=1)))

    # Compute distances from gt to pred
    dists_gt_to_pred = np.zeros(coords_gt.shape[0])
    for i, point_g in enumerate(coords_gt):
        dists_gt_to_pred[i] = np.min(np.sqrt(np.sum((coords_pred - point_g) ** 2, axis=1)))

    all_surface_distances = np.concatenate((dists_pred_to_gt, dists_gt_to_pred))
    if len(all_surface_distances) == 0:  # Should not happen if initial checks pass
        return 0.0

    return np.percentile(all_surface_distances, 95)


# --- DiceLoss (Original, PyTorch-based, no forbidden dependencies) ---
class DiceLoss(nn.Module):  # 原始的dice loss
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


# --- WeightDiceLoss (Modified to handle missing scipy) ---
class WeightDiceLoss(nn.Module):
    def __init__(self, n_classes, edge_weight=True, sigma=2, w0=1, alpha=0.5):
        super(WeightDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.edge_weight = edge_weight
        self.sigma = sigma
        self.w0 = w0
        self.alpha = alpha
        self._scipy_ndimage = None
        if self.edge_weight:
            try:
                from scipy import ndimage as scipy_ndimage
                self._scipy_ndimage = scipy_ndimage
            except ImportError:
                print("Warning: scipy.ndimage not found. WeightDiceLoss edge weighting will be disabled.")
                self.edge_weight = False  # Disable if scipy not found

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _create_edge_weights(self, target_mask):
        batch_size = target_mask.size(0)
        # Assuming target_mask is [B, H, W]
        device = target_mask.device
        weights_list = []

        for b in range(batch_size):
            single_mask_np = target_mask[b].cpu().numpy().astype(np.uint8)
            current_weights = np.ones_like(single_mask_np, dtype=np.float32)

            if self.edge_weight and self._scipy_ndimage:
                edge_map_np = np.zeros_like(single_mask_np, dtype=np.float32)
                for cls in range(self.n_classes):  # Typically n_classes includes background if 0 is background
                    cls_mask = (single_mask_np == cls).astype(np.uint8)
                    if np.sum(cls_mask) == 0:
                        continue

                    eroded = self._scipy_ndimage.binary_erosion(cls_mask, iterations=1)
                    boundary = cls_mask - eroded

                    if np.sum(boundary) == 0 and np.sum(cls_mask) > 0:  # Handle thin structures
                        boundary = cls_mask

                    if np.sum(boundary) > 0:  # only calculate distance if boundary exists
                        distance = self._scipy_ndimage.distance_transform_edt(1 - boundary)
                        cls_weight_values = self.w0 + self.alpha * np.exp(-(distance ** 2) / (2 * self.sigma ** 2))
                        # Apply weights only to pixels of the current class
                        edge_map_np[cls_mask.astype(bool)] = np.maximum(
                            edge_map_np[cls_mask.astype(bool)],
                            cls_weight_values[cls_mask.astype(bool)]
                        )
                    else:  # No boundary found (e.g. isolated pixels or already fully eroded) apply base weight for the class
                        edge_map_np[cls_mask.astype(bool)] = np.maximum(
                            edge_map_np[cls_mask.astype(bool)],
                            self.w0  # Apply base weight w0
                        )

                current_weights = edge_map_np
            weights_list.append(torch.from_numpy(current_weights).to(device))

        return torch.stack(weights_list, dim=0)  # [B, H, W]

    def _dice_loss(self, score, target, weight=None):
        if weight is None:
            weight = torch.ones_like(target)
        target = target.float()  # Ensure target is float
        smooth = 1e-5
        intersect = torch.sum(weight * score * target)
        y_sum = torch.sum(weight * target * target)
        z_sum = torch.sum(weight * score * score)
        dice = (2. * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - dice

    def forward(self, inputs, target, class_weights_list=None, softmax=False):  # Renamed weight to class_weights_list
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target_onehot = self._one_hot_encoder(target)  # [B, C, H, W], target is [B,H,W]

        pixel_wise_weights = self._create_edge_weights(target)  # [B, H, W]

        if class_weights_list is None:
            class_weights_list = [1] * self.n_classes

        assert inputs.size() == target_onehot.size(), 'predict & target shape mismatch'

        total_loss = 0.0
        for i in range(self.n_classes):
            class_input = inputs[:, i]  # [B, H, W]
            class_target = target_onehot[:, i]  # [B, H, W]

            # The pixel_wise_weights are already calculated based on class boundaries.
            # Apply these weights to the loss calculation for the current class.
            # The _dice_loss function will use these weights for all pixels.
            # If a pixel belongs to class_target, its contribution will be weighted.
            # If not, target is 0, so its contribution to numerator and y_sum is 0.
            dice = self._dice_loss(class_input, class_target, pixel_wise_weights)
            total_loss += dice * class_weights_list[i]

        return total_loss / self.n_classes


# --- test_single_volume (Modified to remove scipy, medpy, SimpleITK) ---
def test_single_volume(image_tensor, label_tensor, net, classes, patch_size=[256, 256], test_save_path=None, case=None,
                       z_spacing=1):
    # Ensure input tensors are detached and on CPU for numpy conversion
    image_np = image_tensor.cpu().detach().numpy()
    label_np = label_tensor.cpu().detach().numpy()  # Expected to be [B, H, W] or [B, D, H, W] (integer class labels)

    # Remove batch dimension (assuming B=1)
    if image_np.shape[0] == 1:
        image_np = image_np.squeeze(0)  # Now [C,H,W] or [C,D,H,W]
    if label_np.shape[0] == 1:
        label_np = label_np.squeeze(0)  # Now [H,W] or [D,H,W]

    is_3d = image_np.ndim == 4  # True if image_np is [C,D,H,W]

    # Initialize prediction array to match label's spatial dimensions
    prediction_agg = np.zeros_like(label_np, dtype=np.uint8)

    net.eval()  # Set model to evaluation mode

    if is_3d:
        # image_np is [C, D, H, W], label_np is [D, H, W]
        num_channels, depth, height, width = image_np.shape
        for d_idx in range(depth):
            # Original code took image[0,d], implying single channel from 3D volume for processing
            slice_img_ch0_np = image_np[0, d_idx, :, :]  # [H,W], taking the first channel

            slice_img_tensor = torch.from_numpy(slice_img_ch0_np).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

            if (height, width) != tuple(patch_size):
                resized_slice_tensor = F.interpolate(slice_img_tensor, size=patch_size, mode='bicubic',
                                                     align_corners=False)
            else:
                resized_slice_tensor = slice_img_tensor

            input_cuda = resized_slice_tensor.cuda()  # Assuming net is on CUDA
            with torch.no_grad():
                output_logits = net(input_cuda)  # Expected [1, num_classes, patch_H, patch_W]
                # Argmax directly on logits is fine, softmax won't change argmax
                pred_class_resized = torch.argmax(output_logits, dim=1).squeeze(0).cpu().numpy()  # [patch_H, patch_W]

            if (height, width) != tuple(patch_size):
                pred_class_tensor = torch.from_numpy(pred_class_resized).unsqueeze(0).unsqueeze(0).float()
                resized_pred_class_tensor = F.interpolate(pred_class_tensor, size=(height, width), mode='nearest')
                prediction_agg[d_idx] = resized_pred_class_tensor.squeeze(0).squeeze(0).numpy().astype(np.uint8)
            else:
                prediction_agg[d_idx] = pred_class_resized.astype(np.uint8)
    else:
        # image_np is [C, H, W], label_np is [H, W]
        num_channels, height, width = image_np.shape
        image_tensor_for_net = torch.from_numpy(image_np).unsqueeze(0).float()  # [1,C,H,W]

        if (height, width) != tuple(patch_size):
            image_tensor_for_net = F.interpolate(image_tensor_for_net, size=patch_size, mode='bicubic',
                                                 align_corners=False)

        input_cuda = image_tensor_for_net.cuda()
        with torch.no_grad():
            output_logits = net(input_cuda)  # [1, num_classes, patch_H, patch_W]
            pred_class_resized = torch.argmax(output_logits, dim=1).squeeze(0).cpu().numpy()  # [patch_H, patch_W]

        if (height, width) != tuple(patch_size):
            pred_class_tensor = torch.from_numpy(pred_class_resized).unsqueeze(0).unsqueeze(0).float()
            resized_pred_class_tensor = F.interpolate(pred_class_tensor, size=(height, width), mode='nearest')
            prediction_agg = resized_pred_class_tensor.squeeze(0).squeeze(0).numpy().astype(np.uint8)
        else:
            prediction_agg = pred_class_resized.astype(np.uint8)

    # --- Metric calculation ---
    metric_list = []
    voxelspacing_for_hd95 = None
    if is_3d:
        # Assuming label_np [D,H,W], coords are [idx_d, idx_h, idx_w]
        # z_spacing corresponds to the first dimension (depth)
        voxelspacing_for_hd95 = [z_spacing, 1.0, 1.0]
    else:  # 2D, label_np [H,W]
        voxelspacing_for_hd95 = [1.0, 1.0]  # or actual pixel spacing if known

    for i in range(1, classes):  # Iterate through classes, skipping background (class 0)
        pred_mask_cls = (prediction_agg == i).astype(np.uint8)
        gt_mask_cls = (label_np == i).astype(np.uint8)

        assert pred_mask_cls.shape == gt_mask_cls.shape, \
            f"Class {i}: Shape mismatch - Pred {pred_mask_cls.shape} vs GT {gt_mask_cls.shape}"

        if np.sum(gt_mask_cls) == 0:
            # Original code appends (0,0,0,0,0,0) for empty GT.
            metric_list.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue

        tp = np.sum(pred_mask_cls * gt_mask_cls)
        fp = np.sum(pred_mask_cls * (1 - gt_mask_cls))
        fn = np.sum((1 - pred_mask_cls) * gt_mask_cls)

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        hd95 = 0.0
        if np.sum(pred_mask_cls) > 0:  # Only calculate HD95 if prediction is not empty (GT is not empty by check above)
            try:
                hd95 = hausdorff_distance_95_numpy(pred_mask_cls, gt_mask_cls, voxelspacing_for_hd95)
            except Exception as e_hd:
                print(f"Warning: HD95 calculation failed for class {i}, case {case}: {e_hd}")
                hd95 = 0.0  # Or a specific error marker like np.nan / large value
        # If pred_mask_cls is empty, hd95 remains 0, consistent with original's logic for hd95

        precision = tp / (tp + fp + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)  # Recall
        iou = tp / (tp + fp + fn + 1e-8)  # Jaccard

        vol_pred = np.sum(pred_mask_cls.astype(np.float32))
        vol_gt = np.sum(gt_mask_cls.astype(np.float32))

        vs = 0.0
        # Denominator for VS should be vol_pred + vol_gt. If both are 0, VS is 1.
        # If one is 0 and other is not, VS = 1 - |Vp-Vg|/(Vp+Vg) = 1 - Vg/Vg = 0 (if Vp=0, Vg>0)
        if (vol_pred + vol_gt) > 1e-8:
            vs = 1.0 - (abs(vol_pred - vol_gt) / (vol_pred + vol_gt + 1e-8))
        elif vol_pred == 0 and vol_gt == 0:  # Both are zero (gt_mask_cls > 0 check means this path isn't hit here for vol_gt)
            vs = 1.0  # Perfect similarity if both empty
        vs = np.clip(vs, 0.0, 1.0)

        metric_list.append((dice, hd95, precision, sensitivity, iou, vs))

    # --- Save results ---
    if test_save_path and case:
        os.makedirs(test_save_path, exist_ok=True)
        if is_3d:
            save_path_npy = os.path.join(test_save_path, f"{case}_pred.npy")
            np.save(save_path_npy, prediction_agg)
            # print(f"Saved 3D prediction for {case} as {save_path_npy}")
        else:
            # 2D saving using cv2 (as in original)
            height_pred, width_pred = prediction_agg.shape
            colored_pred_rgb = np.zeros((height_pred, width_pred, 3), dtype=np.uint8)
            color_mapping = {  # RGB colors
                1: [31, 119, 180], 2: [255, 127, 14], 3: [44, 160, 44],
                4: [214, 39, 40], 5: [148, 103, 189], 6: [140, 86, 75],
                7: [227, 119, 194], 8: [188, 189, 34],
            }  # Add more classes if needed
            for cls_val in range(1, classes):  # Iterate up to 'classes-1'
                if cls_val in color_mapping:
                    mask = (prediction_agg == cls_val)
                    colored_pred_rgb[mask] = color_mapping[cls_val]
                else:  # Default color for classes not in map (e.g. white)
                    mask = (prediction_agg == cls_val)
                    colored_pred_rgb[mask] = [255, 255, 255]

            # OpenCV imwrite expects BGR, so convert RGB to BGR
            colored_pred_bgr = cv2.cvtColor(colored_pred_rgb, cv2.COLOR_RGB2BGR)
            save_path_png = os.path.join(test_save_path, f"{case}_pred.png")
            cv2.imwrite(save_path_png, colored_pred_bgr)
            # print(f"Saved 2D color prediction for {case} as {save_path_png}")

    return metric_list

# The other functions (dice_coeff, multiclass_dice_coeff, evaluate) from your
# original code are PyTorch-based and do not have the forbidden dependencies.
# They can be kept as they are if used elsewhere.