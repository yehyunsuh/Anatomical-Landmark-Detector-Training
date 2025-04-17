"""
train.py

Training and evaluation pipeline for anatomical landmark segmentation using U-Net.

Author: Yehyun Suh
"""

import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from data_loader import dataloader
from visualization import overlay_gt_masks, overlay_pred_masks, plot_training_results


def train_model(args, model, device, train_loader, optimizer, loss_fn):
    """
    Train the model for one epoch.

    Args:
        args (Namespace): Configuration arguments.
        model (nn.Module): The model to train.
        device (str): Device to use ('cuda' or 'cpu').
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0

    for images, masks, _, _ in tqdm(train_loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(args, model, device, val_loader, epoch):
    """
    Evaluate the model on validation data.

    Args:
        args (Namespace): Configuration arguments.
        model (nn.Module): Trained model.
        device (str): Device to use.
        val_loader (DataLoader): Validation data loader.
        epoch (int): Current epoch number.

    Returns:
        tuple:
            - avg_loss (float): Average validation loss.
            - dists (Tensor): Landmark prediction errors [N, C].
            - mean_dice (float): Mean Dice score over all landmarks.
    """
    model.eval()
    total_loss = 0
    all_pred_coords = []
    all_gt_coords = []
    all_dice = []

    with torch.no_grad():
        for idx, (images, masks, _, landmarks) in enumerate(tqdm(val_loader, desc="Validation")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = nn.BCEWithLogitsLoss()(outputs, masks)
            total_loss += loss.item()

            # Predict coordinates
            probs = torch.sigmoid(outputs)
            B, C, H, W = probs.shape
            probs_flat = probs.view(B, C, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((B, C, 2), device=device)
            for b in range(B):
                for c in range(C):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, W)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            # Ground truth coordinates
            gt_coords = torch.tensor(landmarks, dtype=torch.float32, device=device)
            if gt_coords.ndim == 2:
                gt_coords = gt_coords.unsqueeze(0)

            all_pred_coords.append(pred_coords)
            all_gt_coords.append(gt_coords)

            if idx == 0:
                overlay_gt_masks(images, masks, pred_coords, gt_coords, epoch, args.epochs, idx)
                overlay_pred_masks(images, outputs, pred_coords, gt_coords, epoch, args.epochs, idx)

            # Dice score computation
            pred_bin = (probs > 0.5).float()
            target_bin = masks
            intersection = (pred_bin * target_bin).sum(dim=(2, 3))
            union = pred_bin.sum(dim=(2, 3)) + target_bin.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            all_dice.append(dice)

    avg_loss = total_loss / len(val_loader)
    all_pred_coords = torch.cat(all_pred_coords, dim=0)
    all_gt_coords = torch.cat(all_gt_coords, dim=0)
    all_dice = torch.cat(all_dice, dim=0)

    dists = torch.norm(all_pred_coords - all_gt_coords, dim=2)
    mean_dist = dists.mean().item()
    mean_dice = all_dice.mean().item()

    evaluate_model.last_dice = all_dice  # Save for later use
    return avg_loss, dists, mean_dice


def train(args, model, device):
    """
    Main training loop with dynamic erosion/dilation and weighted loss adjustment.

    Args:
        args (Namespace): Configuration arguments.
        model (nn.Module): Model to train.
        device (str): Computation device.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_mean_error = float("inf")

    # History tracker
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "mean_landmark_error": [],
        "landmark_errors": {str(c): [] for c in range(args.n_landmarks)},
        "mean_dice": [],
        "dice_scores": {str(c): [] for c in range(args.n_landmarks)},
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Recalculate loss weighting every erosion_freq epochs
        if epoch % args.erosion_freq == 0:
            if epoch != 0:
                args.dilation_iters = max(args.dilation_iters - args.erosion_iters, 1)

            image_size = args.image_resize ** 2
            n_dilated_pixels = 1 + 2 * args.dilation_iters * (args.dilation_iters + 1)
            weight_ratio = (image_size * 100 / n_dilated_pixels) / (image_size * 100 / (image_size - n_dilated_pixels))
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_ratio).to(device))

            train_loader, val_loader = dataloader(args)
            print(f"Loss weight: {weight_ratio:.4f}")

        train_loss = train_model(args, model, device, train_loader, optimizer, loss_fn)
        val_loss, dists, mean_dice = evaluate_model(args, model, device, val_loader, epoch)
        mean_dist = dists.mean().item()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean Dist: {mean_dist:.4f} | Mean Dice: {mean_dice:.4f}")

        if mean_dist < best_mean_error:
            best_mean_error = mean_dist
            torch.save(model.state_dict(), "weight/best_model.pth")
            print("✅ Saved new best model!")

        # Log epoch stats
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mean_landmark_error"].append(mean_dist)
        history["mean_dice"].append(mean_dice)

        for c in range(dists.shape[1]):
            history["landmark_errors"][str(c)].append(dists[:, c].mean().item())
            history["dice_scores"][str(c)].append(evaluate_model.last_dice[:, c].mean().item())

    # Final plot
    plot_training_results(history)

    # Build rows for CSV
    rows = []
    for i, epoch in enumerate(history["epoch"]):
        row = [
            epoch,
            history["train_loss"][i],
            history["val_loss"][i],
            history["mean_landmark_error"][i],
        ]
        # Append per-landmark error
        for c in range(args.n_landmarks):
            row.append(history["landmark_errors"][str(c)][i])
        
        # Append mean Dice
        row.append(history["mean_dice"][i])
        
        # Append per-landmark Dice
        for c in range(args.n_landmarks):
            row.append(history["dice_scores"][str(c)][i])

        rows.append(row)

    # Create column headers
    columns = ["epoch", "train_loss", "val_loss", "mean_dist"]
    columns += [f"landmark{c+1}_dist" for c in range(args.n_landmarks)]
    columns += ["mean_dice"]
    columns += [f"landmark{c+1}_dice" for c in range(args.n_landmarks)]

    # Create DataFrame and save
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("train_results/training_log.csv", index=False)
    print("📄 Saved training log to train_results/training_log.csv")