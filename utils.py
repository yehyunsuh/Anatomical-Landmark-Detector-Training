"""
utils.py

Utility functions for model training and reproducibility.

Author: Yehyun Suh
"""

import torch
import random
import numpy as np


def customize_seed(seed):
    """
    Sets seeds across libraries to ensure reproducible results.

    Args:
        seed (int): The seed value to be used for all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Uncomment below if using multi-GPU setup
    # torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)