import os, random, json
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def collate_video_batch(batch):
    return {k: torch.stack([x[k] for x in batch]) if torch.is_tensor(batch[0][k]) else [x[k] for x in batch] for k in batch[0]}
