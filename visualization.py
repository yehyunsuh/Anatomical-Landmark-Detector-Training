"""
visualization.py

Visualization utilities for overlaying predicted/ground truth landmark masks
and plotting training statistics for segmentation performance.

Author: Yehyun Suh
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def overlay_gt_masks(images, masks, pred_coords, gt_coords, epoch, total_epoch, idx):
    """
    Overlay ground truth masks and landmarks on the image and save as visualization.

    Args:
        images (Tensor): Batch of images [B, 3, H, W].
        masks (Tensor): Batch of ground truth masks [B, C, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch number.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.
    """
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8).copy()

        # Convert multi-channel mask to binary
        gt_mask = masks[b].sum(0).cpu().numpy()
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        # Blend image and GT mask
        overlay = cv2.addWeighted(img, 0.7, gt_mask_rgb, 0.3, 0)

        # Draw GT and predicted landmark circles
        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)   # Red: Predicted
            cv2.circle(overlay, (gx, gy), 4, (0, 255, 0), -1)   # Green: Ground Truth

        # Save visualization
        if epoch % 10 == 0 or epoch == total_epoch - 1:
            os.makedirs(f"visualization/Epoch{epoch}", exist_ok=True)
            cv2.imwrite(f"visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_overlay_gt.png", overlay)
        cv2.imwrite(f"visualization/Batch{idx}_overlay_gt.png", overlay)


def overlay_pred_masks(images, outputs, pred_coords, gt_coords, epoch, total_epoch, idx):
    """
    Overlay predicted masks and landmarks per landmark channel on the image.

    Args:
        images (Tensor): Batch of images [B, 3, H, W].
        outputs (Tensor): Raw model outputs [B, C, H, W].
        pred_coords (Tensor): Predicted landmark coordinates [B, C, 2].
        gt_coords (Tensor): Ground truth landmark coordinates [B, C, 2].
        epoch (int): Current epoch number.
        total_epoch (int): Total number of epochs.
        idx (int): Batch index.
    """
    for b in range(images.shape[0]):
        pred = pred_coords[b].cpu().numpy()
        gt = gt_coords[b].cpu().numpy()

        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8).copy()

        C = outputs.shape[1]
        for c in range(C):
            # Create predicted mask
            mask = (torch.sigmoid(outputs[b, c]) > 0.5).float().cpu().numpy().astype(np.uint8) * 255
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Blend with image
            overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)

            # Draw landmark circles
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)  # Red = predicted
            cv2.circle(overlay, (gx, gy), 4, (0, 255, 0), -1)  # Green = GT

            # Save overlay
            if epoch % 10 == 0 or epoch == total_epoch - 1:
                os.makedirs(f"visualization/Epoch{epoch}", exist_ok=True)
                cv2.imwrite(f"visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_Landmark{c}.png", overlay)
            cv2.imwrite(f"visualization/Batch{idx}_Landmark{c}.png", overlay)


def plot_training_results(history):
    """
    Plot and save training curves for loss, landmark error, and Dice scores.

    Args:
        history (dict): Dictionary containing metric history across epochs.
    """
    os.makedirs("graph", exist_ok=True)

    # Train/val loss
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph/loss_curve.png")

    # Mean landmark error
    plt.figure()
    plt.plot(history["epoch"], history["mean_landmark_error"], label="Mean Landmark Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.grid(True)
    plt.savefig("graph/mean_landmark_error.png")

    # Per-landmark error
    plt.figure()
    for k, v in history["landmark_errors"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph/per_landmark_error.png")

    # Mean Dice score
    plt.figure()
    plt.plot(history["epoch"], history["mean_dice"], label="Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.savefig("graph/mean_dice_score.png")

    # Per-landmark Dice score
    plt.figure()
    for k, v in history["dice_scores"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph/per_landmark_dice.png")

    plt.close("all")