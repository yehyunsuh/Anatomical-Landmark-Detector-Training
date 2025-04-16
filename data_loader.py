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
    def __init__(self, csv_path, image_dir, transform=None, n_landmarks=None, dilation_iters=None):
        self.image_dir = image_dir
        self.samples = []
        self.transform = transform
        self.n_landmarks = n_landmarks
        self.dilation_iters = dilation_iters

        # Parse CSV
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                image_name = row[0]
                coords = list(map(int, row[4:]))
                assert len(coords) == 2 * n_landmarks
                landmarks = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                self.samples.append((image_name, landmarks))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, landmarks = self.samples[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        max_side = max(h, w)

        transform = A.Compose([
            A.PadIfNeeded(min_height=max_side, min_width=max_side),
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        transformed = transform(image=image, keypoints=landmarks)
        image = transformed['image']  # [3, H, W]
        new_landmarks = transformed['keypoints']

        H, W = image.shape[1:]  # after resizing
        masks = np.zeros((self.n_landmarks, H, W), dtype=np.uint8)
        for k, (x, y) in enumerate(new_landmarks):
            x = int(round(x))
            y = int(round(y))
            if 0 <= y < H and 0 <= x < W:
                masks[k, y, x] = 1
                masks[k] = binary_dilation(masks[k], iterations=self.dilation_iters).astype(np.uint8)

        mask = torch.from_numpy(masks).float()

        return image, mask, image_name, new_landmarks


def dataloader(args):
    dataset = SegmentationDataset(
        csv_path=os.path.join(args.label_dir, args.train_csv_file),
        image_dir=args.train_image_dir,
        transform=None,
        n_landmarks=args.n_landmarks,
        dilation_iters=args.dilation_iters,
    )

    # Split lengths
    train_size = int(0.8 * len(dataset))
    val_size = int(0.2 * len(dataset))

    # Reproducible splitting
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")

    # # Example: visualize a batch
    # for batch_idx, (images, labels, img_name) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}: {img_name}")
    #     print(f"  Image shape: {images.shape}")
    #     print(f"  Label shape: {labels.shape}")
        
    #     # Convert image tensor to uint8 numpy [H, W, 3]
    #     img_np = images[0].permute(1, 2, 0).cpu().numpy()  # [-1, 1]
    #     img_np = ((img_np * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    #     cv2.imwrite('tmp/test.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    #     # Convert landmark masks to numpy
    #     masks_np = labels[0].cpu().numpy()  # [n_landmarks, H, W]

    #     # For each mask, save raw and overlaid mask
    #     for i in range(args.n_landmarks):
    #         mask = (masks_np[i] * 255).astype(np.uint8)  # binary mask to 0 or 255

    #         # Save raw mask
    #         cv2.imwrite(f"tmp/test_mask{i}.jpg", mask)

    #         # Convert mask to 3-channel color (e.g., red overlay)
    #         color_mask = np.zeros_like(img_np)
    #         color_mask[:, :, 2] = mask  # Red channel

    #         # Overlay with original image (blend)
    #         overlay = cv2.addWeighted(img_np, 1.0, color_mask, 0.5, 0)

    #         # Save the overlay
    #         cv2.imwrite(f"tmp/test_overlay_mask{i}.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    #     exit()

    return train_loader, val_loader