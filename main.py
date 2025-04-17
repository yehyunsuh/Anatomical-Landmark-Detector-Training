"""
main.py

Training pipeline for anatomical landmark detection using U-Net.

This script parses command-line arguments, prepares the training environment,
loads the model, and initiates the training loop with configurable data and
hyperparameters.

Author: Yehyun Suh  
Date: 2025-04-15  
Copyright: (c) 2025 Yehyun Suh

Example:
    python main.py \
        --train_image_dir ./data/train_images \
        --label_dir ./data/labels \
        --train_csv_file train_annotation.csv \
        --image_resize 512 \
        --n_landmarks 2 \
        --batch_size 8 \
        --epochs 350 \
        --dilation_iters 65 \
        --erosion_freq 50 \
        --erosion_iters 10 \
        --gif
"""

import os
import argparse
import torch

from utils import customize_seed
from model import UNet
from train import train


def main(args):
    """
    Main function that initializes and trains the model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNet(args.n_landmarks, device)

    # Start training
    train(args, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script for anatomical landmark segmentation with U-Net."
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducibility"
    )

    # Data paths
    parser.add_argument(
        "--train_image_dir", type=str, default="./data/train_images",
        help="Directory containing training images"
    )
    parser.add_argument(
        "--label_dir", type=str, default="./data/labels",
        help="Directory containing ground truth annotation CSVs"
    )
    parser.add_argument(
        "--train_csv_file", type=str, default="train_annotation.csv",
        help="CSV file containing training annotations"
    )

    # Image/label settings
    parser.add_argument(
        "--image_resize", type=int, default=512,
        help="Target image size after resizing (must be divisible by 32)"
    )
    parser.add_argument(
        "--n_landmarks", type=int, required=True,
        help="Number of landmarks per image"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=350,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--dilation_iters", type=int, default=65,
        help="Number of iterations for binary dilation"
    )
    parser.add_argument(
        "--erosion_freq", type=int, default=50,
        help="Apply erosion every N epochs"
    )
    parser.add_argument(
        "--erosion_iters", type=int, default=10,
        help="Number of iterations for binary erosion"
    )

    # Visualization options
    parser.add_argument(
        "--gif", action="store_true",
        help="Enable GIF creation of training visuals"
    )

    args = parser.parse_args()

    # Fix randomness
    customize_seed(args.seed)

    # Create necessary directories
    os.makedirs("visualization", exist_ok=True)
    os.makedirs("graph", exist_ok=True)
    os.makedirs("weight", exist_ok=True)
    os.makedirs("train_results", exist_ok=True)

    main(args)