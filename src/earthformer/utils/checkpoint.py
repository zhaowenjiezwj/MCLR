import os
import warnings
from typing import Callable, Dict, Optional, Union, List
import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load


def average_checkpoints(checkpoint_paths: List[str] = None):
    if len(checkpoint_paths) > 1:
        avg_state_dict = {}
        avg_counts = {}
        for per_path in checkpoint_paths:
            state_dict = torch.load(per_path, map_location=torch.device("cpu"))["state_dict"]
            for k, v in state_dict.items():
                if k not in avg_state_dict:
                    avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                    avg_counts[k] = 1
                else:
                    avg_state_dict[k] += v.to(dtype=torch.float64)
                    avg_counts[k] += 1
            del state_dict

        for k, v in avg_state_dict.items():
            v.div_(avg_counts[k])

        # convert to float32.
        float32_info = torch.finfo(torch.float32)
        for k in avg_state_dict:
            avg_state_dict[k].clamp_(float32_info.min, float32_info.max).to(dtype=torch.float32)
    else:
        avg_state_dict = torch.load(checkpoint_paths[0], map_location=torch.device("cpu"))["state_dict"]

    return avg_state_dict

def average_pl_checkpoints(pl_checkpoint_paths: List[str] = None, delete_prefix_len: int = len("")):
    if len(pl_checkpoint_paths) > 1:
        avg_state_dict = {}
        avg_counts = {}
        for per_path in pl_checkpoint_paths:
            state_dict = pl_ckpt_to_pytorch_state_dict(per_path,
                                                       map_location=torch.device("cpu"),
                                                       delete_prefix_len=delete_prefix_len)
            for k, v in state_dict.items():
                if k not in avg_state_dict:
                    avg_state_dict[k] = v.clone().to(dtype=torch.float64)
                    avg_counts[k] = 1
                else:
                    avg_state_dict[k] += v.to(dtype=torch.float64)
                    avg_counts[k] += 1
            del state_dict

        for k, v in avg_state_dict.items():
            v.div_(avg_counts[k])

        # convert to float32.
        float32_info = torch.finfo(torch.float32)
        for k in avg_state_dict:
            avg_state_dict[k].clamp_(float32_info.min, float32_info.max).to(dtype=torch.float32)
    else:
        avg_state_dict = pl_ckpt_to_pytorch_state_dict(pl_checkpoint_paths[0],
                                                       map_location=torch.device("cpu"),
                                                       delete_prefix_len=delete_prefix_len)

    return avg_state_dict

def pl_ckpt_to_pytorch_state_dict(
        checkpoint_path: str,
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        delete_prefix_len: int = len("")):
    if map_location is not None:
        checkpoint = pl_load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)
    pl_ckpt_state_dict = checkpoint["state_dict"]
    pytorch_state_dict = {key[delete_prefix_len:]: val
                          for key, val in pl_ckpt_state_dict.items()}
    return pytorch_state_dict


def s3_download_pretrained_ckpt(ckpt_name, save_dir=None, exist_ok=False):
    if save_dir is None:
        from ..config import cfg
        save_dir = cfg.pretrained_checkpoints_dir
    if os.path.exists(os.path.join(save_dir, ckpt_name)) and not exist_ok:
        warnings.warn(f"Checkpoint file {os.path.join(save_dir, ckpt_name)} already exists!")
    else:
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"aws s3 cp --no-sign-request s3://earthformer/pretrained_checkpoints/{ckpt_name} "
                  f"{save_dir}")
