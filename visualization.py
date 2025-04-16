import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def overlay_gt_masks(images, masks, pred_coords, gt_coords, epoch, total_epoch, idx):
    for b in range(images.shape[0]):  # visualize every image in the batch
        pred = pred_coords[b].cpu().numpy()  # [C, 2]
        gt = gt_coords[b].cpu().numpy()  # [C, 2]
        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8).copy()

        # Convert the GT mask to a 2D binary map by summing across landmarks
        gt_mask = masks[b].sum(0).cpu().numpy()  # shape: [H, W]
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255  # binary mask

        # Convert to 3-channel for overlay
        gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)

        # Blend image and mask
        overlay = cv2.addWeighted(img, 0.7, gt_mask_rgb, 0.3, 0)

        # Draw landmarks again on top of the overlay
        for c in range(pred.shape[0]):
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)
            cv2.circle(overlay, (gx, gy), 4, (0, 255, 0), -1)

        # Save the overlay
        if epoch % 10 == 0 or epoch == total_epoch - 1:
            os.makedirs(f"visualization/Epoch{epoch}", exist_ok=True)
            cv2.imwrite(f"visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_overlay_gt.png", overlay)
        cv2.imwrite(f"visualization/Batch{idx}_overlay_gt.png", overlay)


def overlay_pred_masks(images, outputs, pred_coords, gt_coords, epoch, total_epoch, idx):
    for b in range(images.shape[0]):  # visualize every image in the batch
        pred = pred_coords[b].cpu().numpy()  # [C, 2]
        gt = gt_coords[b].cpu().numpy()      # [C, 2]
        img = images[b].cpu().permute(1, 2, 0).numpy()
        img = ((img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8).copy()

        C = outputs.shape[1]
        for c in range(C):
            # Extract predicted mask for landmark c
            mask = (torch.sigmoid(outputs[b, c]) > 0.5).float().cpu().numpy().astype(np.uint8) * 255  # [H, W]
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Blend with original image
            overlay = cv2.addWeighted(img, 0.7, mask_rgb, 0.3, 0)

            # Draw landmark points
            px, py = int(pred[c, 0]), int(pred[c, 1])
            gx, gy = int(gt[c, 0]), int(gt[c, 1])

            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)  # red = predicted
            cv2.circle(overlay, (gx, gy), 4, (0, 255, 0), -1)  # green = GT

            # Save overlay per landmark
            if epoch % 10 == 0 or epoch == total_epoch - 1:
                os.makedirs(f"visualization/Epoch{epoch}", exist_ok=True)
                cv2.imwrite(f"visualization/Epoch{epoch}/Epoch{epoch}_Batch{idx}_Landmark{c}.png", overlay)
            cv2.imwrite(f"visualization/Batch{idx}_Landmark{c}.png", overlay)


def plot_training_results(history):
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph/loss_curve.png")

    plt.figure()
    plt.plot(history["epoch"], history["mean_landmark_error"], label="Mean Landmark Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.grid(True)
    plt.savefig("graph/mean_landmark_error.png")

    plt.figure()
    for k, v in history["landmark_errors"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Error (px)")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph/per_landmark_error.png")

    plt.figure()
    plt.plot(history["epoch"], history["mean_dice"], label="Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.savefig("graph/mean_dice_score.png")

    plt.figure()
    for k, v in history["dice_scores"].items():
        plt.plot(history["epoch"], v, label=f"Landmark {k}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph/per_landmark_dice.png")

    plt.close("all")