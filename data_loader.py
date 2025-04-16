"""
data_loader.py

Dataset and dataloader utilities for anatomical landmark segmentation.

Author: Yehyun Suh
"""

import os
import csv
import cv2
import torch
import numpy as np
import albumentations as A

from scipy.ndimage import binary_dilation
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split


class SegmentationDataset(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    Each sample includes an RGB image and a multi-channel binary mask,
    where each channel corresponds to a dilated landmark point.
    """

    def __init__(self, csv_path, image_dir, transform=None, n_landmarks=None, dilation_iters=None):
        """
        Initializes the dataset by parsing CSV annotations and storing image/landmark paths.

        Args:
            csv_path (str): Path to the annotation CSV file.
            image_dir (str): Directory containing input images.
            transform (albumentations.Compose, optional): Image transform pipeline.
            n_landmarks (int): Number of landmarks per image.
            dilation_iters (int): Number of dilation iterations for landmark masks.
        """
        self.image_dir = image_dir
        self.samples = []
        self.transform = transform
        self.n_landmarks = n_landmarks
        self.dilation_iters = dilation_iters

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                image_name = row[0]
                coords = list(map(int, row[4:]))
                assert len(coords) == 2 * n_landmarks, "Mismatch in number of landmark coordinates"
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Load and convert image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        max_side = max(h, w)

        # Apply resizing and normalization
        transform = A.Compose([
            A.PadIfNeeded(min_height=max_side, min_width=max_side),
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        transformed = transform(image=image, keypoints=landmarks)
        image = transformed['image']  # Tensor: [3, H, W]
        new_landmarks = transformed['keypoints']

        # Create mask with binary dilation for each landmark
        H, W = image.shape[1:]
        masks = np.zeros((self.n_landmarks, H, W), dtype=np.uint8)

        for k, (x, y) in enumerate(new_landmarks):
            x = int(round(x))
            y = int(round(y))
            if 0 <= y < H and 0 <= x < W:
                masks[k, y, x] = 1
                masks[k] = binary_dilation(masks[k], iterations=self.dilation_iters).astype(np.uint8)

        mask = torch.from_numpy(masks).float()  # Shape: [n_landmarks, H, W]

        return image, mask, image_name, new_landmarks


def dataloader(args):
    """
    Constructs and returns PyTorch dataloaders for training and validation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing all config.

    Returns:
        tuple: (train_loader, val_loader)
    """
    dataset = SegmentationDataset(
        csv_path=os.path.join(args.label_dir, args.train_csv_file),
        image_dir=args.train_image_dir,
        transform=None,
        n_landmarks=args.n_landmarks,
        dilation_iters=args.dilation_iters,
    )

    # Train-validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")

    return train_loader, val_loader