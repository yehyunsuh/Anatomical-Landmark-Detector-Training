import torch.nn as nn
import segmentation_models_pytorch as smp


def UNet(n_landmarks, DEVICE):
    print("---------- Loading  Model ----------")

    model = smp.Unet(
        encoder_name     = 'resnet101', 
        encoder_weights  = 'imagenet', 
        classes          = n_landmarks, 
        activation       = 'sigmoid',
    )
    print("---------- Model Loaded ----------")

    # Erase Sigmoid
    model.segmentation_head = nn.Sequential(*list(model.segmentation_head.children())[:-1])
    
    return model.to(DEVICE)