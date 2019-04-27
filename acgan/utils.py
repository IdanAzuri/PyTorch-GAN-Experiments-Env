
import os
from io import BytesIO
import numpy as np
import re
import scipy.misc
# import tensorflow as tf
import torch



def load_saved_model(path, model, optimizer):
    latest_path = find_latest(path)
    if latest_path is None:
        return 0, model, optimizer

    checkpoint = torch.load(latest_path)

    step_count = checkpoint['step_count']
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Load checkpoints...! {latest_path}")
    return step_count, model, optimizer


def find_latest(find_path):
    sorted_path = get_sorted_path(find_path)
    if len(sorted_path) == 0:
        return None

    return sorted_path[-1]


def save_checkpoint(step, path, model, optimizer, max_to_keep=10):
    sorted_path = get_sorted_path(path)
    for i in range(len(sorted_path) - max_to_keep):
        os.remove(sorted_path[i])

    full_path = path + f"-{step}.pkl"
    torch.save({
        "step_count": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, full_path)
    print(f"Save checkpoints...! {full_path}")


def get_sorted_path(find_path):
    dir_path = os.path.dirname(find_path)
    base_name = os.path.basename(find_path)

    paths = []
    for root, dirs, files in os.walk(dir_path):
        for f_name in files:
            if f_name.startswith(base_name) and f_name.endswith(".pkl"):
                paths.append(os.path.join(root, f_name))

    return sorted(paths, key=lambda x: int(re.findall("\d+", os.path.basename(x))[0]))


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



