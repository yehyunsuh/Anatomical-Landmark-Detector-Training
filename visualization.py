"""
visualization.py

Visualization utilities for overlaying predicted/ground truth landmark masks
and plotting training statistics for segmentation performance.

Author: Yehyun Suh  
Date: 2025-04-15
"""

import os
import cv2
import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def overlay_gt_masks(args, images, masks, pred_coords, gt_coords, epoch, total_epoch, idx):
    """
    Overlay ground truth masks and landmarks on the image and save as visualization.

    Args:
        args (Namespace): Configuration arguments.
        images (Tensor): Batch of images [B, 3, H, W].
        masks (Tensor): Batch of ground truth masks [B, C, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch number.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.

    Returns:
        ndarray: Overlaid image with ground truth mask and landmarks.
    """
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        gt_mask = masks[b].sum(0).cpu().numpy()
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(img, 0.7, gt_mask_rgb, 0.3, 0)

        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(overlay, (gx, gy), 4, (255, 0, 0), -1)

        if epoch % 10 == 0 or epoch == total_epoch - 1:
            os.makedirs(f"{args.experiment_name}/visualization/Epoch{epoch}", exist_ok=True)
            cv2.imwrite(
                f"{args.experiment_name}/visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_overlay_gt.png", overlay
            )
        cv2.imwrite(f"{args.experiment_name}/visualization/Batch{idx}_overlay_gt.png", overlay)

        return overlay


def overlay_pred_masks(args, images, outputs, pred_coords, gt_coords, epoch, total_epoch, idx):
    """
    Overlay predicted masks and landmarks per landmark channel on the image.

    Args:
        args (Namespace): Configuration arguments.
        images (Tensor): Batch of input images [B, 3, H, W].
        outputs (Tensor): Raw model outputs [B, C, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.

    Returns:
        list: List of overlaid images for each landmark.
    """
    overlay_list = []

    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        C = outputs.shape[1]
        for c in range(C):
            mask = (torch.sigmoid(outputs[b, c]) > 0.5).float().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)

            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(overlay, (gx, gy), 4, (255, 0, 0), -1)

            if epoch % 10 == 0 or epoch == total_epoch - 1:
                os.makedirs(f"{args.experiment_name}/visualization/Epoch{epoch}", exist_ok=True)
                cv2.imwrite(
                    f"{args.experiment_name}/visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_Landmark{c}.png", overlay
                )
            cv2.imwrite(f"{args.experiment_name}/visualization/Batch{idx}_Landmark{c}.png", overlay)

            overlay_list.append(overlay)

    return overlay_list


def overlay_pred_coords(args, images, pred_coords, gt_coords, epoch, total_epoch, idx):
    """
    Overlay only landmark coordinates (no mask) on the image.

    Args:
        args (Namespace): Configuration arguments.
        images (Tensor): Batch of input images [B, 3, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.

    Returns:
        ndarray: Image with overlaid coordinates.
    """
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) +
               np.array([0.485, 0.456, 0.406])) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8).copy()

        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(img, (gx, gy), 4, (255, 0, 0), -1)

        if epoch % 10 == 0 or epoch == total_epoch - 1:
            os.makedirs(f"{args.experiment_name}/visualization/Epoch{epoch}", exist_ok=True)
            cv2.imwrite(
                f"{args.experiment_name}/visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_overlay_pred.png", img
            )
        cv2.imwrite(f"{args.experiment_name}/visualization/Batch{idx}_overlay_pred.png", img)

        return img


def create_gif(args, gt_mask_w_coords_image_list, pred_mask_w_coords_image_list_list, coords_image_list):
    """
    Create and save animated GIFs to visualize model predictions over epochs.

    Args:
        args (Namespace): Configuration arguments.
        gt_mask_w_coords_image_list (list): Ground truth mask overlays.
        pred_mask_w_coords_image_list_list (list of list): List of predicted mask overlays per landmark.
        coords_image_list (list): Coordinate-only overlays.
    """
    def convert_to_numpy(image_list):
        converted = []
        for img in image_list:
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            converted.append(img)
        return converted

    gt_mask_frames = convert_to_numpy(gt_mask_w_coords_image_list)

    # pred_mask_frames = []
    # for i in range(len(pred_mask_w_coords_image_list_list[0])):
    #     per_landmark = [frame_list[i] for frame_list in pred_mask_w_coords_image_list_list]
    #     pred_mask_frames.append(convert_to_numpy(per_landmark))

    coords_frames = convert_to_numpy(coords_image_list)

    imageio.mimsave(f"{args.experiment_name}/train_results/gt_mask_with_coords.gif", gt_mask_frames, fps=10)
    # for i, frames in enumerate(pred_mask_frames):
    #     imageio.mimsave(f"{args.experiment_name}/train_results/pred_mask_with_coords_{i}.gif", frames, fps=10)
    imageio.mimsave(f"{args.experiment_name}/train_results/pred_coords_only.gif", coords_frames, fps=10)

    print("🖼️ Saved training progress GIFs to train_results/")


def plot_training_results(args, history):
    """
    Plot and save training curves for loss, error, and Dice score.

    Args:
        args (Namespace): Configuration arguments.
        history (dict): Training history with metrics over epochs.
    """
    # Loss curve
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.experiment_name}/graph/loss_curve.png")

    # Mean landmark error
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["mean_landmark_error"], label="Mean Landmark Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.grid(True)
    plt.savefig(f"{args.experiment_name}/graph/mean_landmark_error.png")

    # Log-scale version
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["mean_landmark_error"], label="Mean Landmark Error")
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(f"{args.experiment_name}/graph/mean_landmark_error_log.png")

    # Per-landmark error
    plt.figure(figsize=(12, 8))
    for k, v in history["landmark_errors"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.experiment_name}/graph/per_landmark_error.png")

    # Log-scale per-landmark error
    plt.figure(figsize=(12, 8))
    for k, v in history["landmark_errors"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$'))
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig(f"{args.experiment_name}/graph/per_landmark_error_log.png")

    # Mean Dice score
    plt.figure(figsize=(12, 8))
    plt.plot(history["epoch"], history["mean_dice"], label="Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.savefig(f"{args.experiment_name}/graph/mean_dice_score.png")

    # Per-landmark Dice scores
    plt.figure(figsize=(12, 8))
    for k, v in history["dice_scores"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.experiment_name}/graph/per_landmark_dice.png")

    plt.close("all")