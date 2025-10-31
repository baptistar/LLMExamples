import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)  # Fix seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)         # if using CUDA
    torch.cuda.manual_seed_all(seed)     # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

