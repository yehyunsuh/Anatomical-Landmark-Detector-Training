"""
model.py

Defines the U-Net segmentation model for anatomical landmark detection.

Author: Yehyun Suh
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


def UNet(n_landmarks, device):
    """
    Constructs and returns a U-Net model using a ResNet-101 encoder,
    configured for anatomical landmark detection.

    Args:
        n_landmarks (int): Number of landmarks (output channels).
        device (str): Device to move the model to ('cuda' or 'cpu').

    Returns:
        nn.Module: The U-Net model ready for training.
    """
    print("---------- Loading Model ----------")

    model = smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        classes=n_landmarks,
        activation='sigmoid',  # This will be removed below
    )

    print("---------- Model Loaded ----------")

    # Remove the final sigmoid activation from the segmentation head
    # so loss functions like BCEWithLogitsLoss can be used instead
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])

    return model.to(device)