"""
main.py
"""

import os
import torch
import argparse

from utils import customize_seed
from model import UNet
from train import train


def main(args):
    # Load model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    model = UNet(args.n_landmarks, DEVICE)

    # Train the model
    train(args, model, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for the project.")

    # Arguments for reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    # Arguments for data
    parser.add_argument('--train_image_dir', type=str, default='./data/train_images', help='Directory for images')
    parser.add_argument('--test_image_dir', type=str, default='./data/test_images', help='Directory for images')
    parser.add_argument('--label_dir', type=str, default='./data/labels', help='Directory for labels (annotations)')
    parser.add_argument('--train_csv_file', type=str, default='train_annotation.csv', help='Name of the CSV file with image and label names')
    parser.add_argument('--test_csv_file', type=str, default='test_annotation.csv', help='Name of the CSV file with image and label names')

    parser.add_argument('--image_resize', type=int, default=512, help='Size of the images after resizing, it should be divisible by 32')
    parser.add_argument('--n_landmarks', type=int, default=2, required=True, help='Number of landmarks')
    
    # Arguments for training
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=350, help='Number of epochs for training')

    parser.add_argument('--dilation_iters', type=int, default=65, help='Number of iterations for binary dilation on masks')
    parser.add_argument('--erosion_freq', type=int, default=50, help='Frequency of erosion during training')
    parser.add_argument('--erosion_iters', type=int, default=10, help='Number of iterations for binary erosion on masks')
    
    args = parser.parse_args()
    
    # Fix seed for reproducibility
    customize_seed(args.seed)

    # Create directories for visualization
    os.makedirs("visualization", exist_ok=True)
    os.makedirs("graph", exist_ok=True)
    os.makedirs("weight", exist_ok=True)

    main(args)