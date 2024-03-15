import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import DataLoader
import glob
import numpy as np
import os
import random
import copy


def generate_segment_indices(
    videopath1, videopath2, num_input_frames=10, filename_tmpl="{:08d}.png"
):
    """generate segment function
    Args:
        videopath1,2 (str): input directory which contains sequential frames
        filename_tmpl (str): template which represents sequential frames
    Returns:
        Tensor, Tensor: Output sequence with shape (t, c, h, w)
    """
    seq_length = len(glob.glob(f"{videopath1}/*.png"))
    seq_length2 = len(glob.glob(f"{videopath2}/*.png"))

    if seq_length != seq_length2:
        raise ValueError(
            f"{videopath1} and {videopath2} must have same number of frames \n but they have {seq_length} and {seq_length2}"
        )
    if num_input_frames > seq_length:
        raise ValueError(
            f"num_input_frames{num_input_frames} must be greater than frames in {videopath1} \n and {videopath2}"
        )

    start_frame_idx = np.random.randint(0, seq_length - num_input_frames + 1)
    end_frame_idx = start_frame_idx + num_input_frames
    segment1 = [
        read_image(os.path.join(videopath1, filename_tmpl.format(i))) / 255.0
        for i in range(start_frame_idx, end_frame_idx)
    ]
    segment2 = [
        read_image(os.path.join(videopath2, filename_tmpl.format(i))) / 255.0
        for i in range(start_frame_idx, end_frame_idx)
    ]
    return torch.stack(segment1), torch.stack(segment2)


def pair_random_crop_seq(hr_seq, lr_seq, patch_size, scale_factor=4):
    """crop image pair for data augment
    Args:
        hr (Tensor): hr images with shape (t, c, 4h, 4w).
        lr (Tensor): lr images with shape (t, c, h, w).
        patch_size (int): the size of cropped image
    Returns:
        Tensor, Tensor: cropped images(hr,lr)
    """
    seq_lenght = lr_seq.size(dim=0)
    gt_transformed = torch.empty(
        seq_lenght, 3, patch_size * scale_factor, patch_size * scale_factor
    )
    lq_transformed = torch.empty(seq_lenght, 3, patch_size, patch_size)
    i, j, h, w = T.RandomCrop.get_params(
        lr_seq[0], output_size=(patch_size, patch_size)
    )
    gt_transformed = T.functional.crop(
        hr_seq, i * scale_factor, j * scale_factor, h * scale_factor, w * scale_factor
    )
    lq_transformed = T.functional.crop(lr_seq, i, j, h, w)
    return gt_transformed, lq_transformed


def pair_random_flip_seq(sequence1, sequence2, p=0.5, horizontal=True, vertical=True):
    """flip image pair for data augment
    Args:
        sequence1 (Tensor): images with shape (t, c, h, w).
        sequence2 (Tensor): images with shape (t, c, h, w).
        p (float): probability of the image being flipped.
            Default: 0.5
        horizontal (bool): Store `False` when don't flip horizontal
            Default: `True`.
        vertical (bool): Store `False` when don't flip vertical
            Default: `True`.
    Returns:
        Tensor, Tensor: cropped images
    """
    T_length = sequence1.size(dim=0)
    # Random horizontal flipping
    hfliped1 = sequence1.clone()
    hfliped2 = sequence2.clone()
    if horizontal and random.random() > p:
        hfliped1 = T.functional.hflip(sequence1)
        hfliped2 = T.functional.hflip(sequence2)

    # Random vertical flipping
    vfliped1 = hfliped1.clone()
    vfliped2 = hfliped2.clone()
    if vertical and random.random() > p:
        vfliped1 = T.functional.vflip(hfliped1)
        vfliped2 = T.functional.vflip(hfliped2)
    return vfliped1, vfliped2


def pair_random_transposeHW_seq(sequence1, sequence2, p=0.5):
    """crop image pair for data augment
    Args:
        sequence1 (Tensor): images with shape (t, c, h, w).
        sequence2 (Tensor): images with shape (t, c, h, w).
        p (float): probability of the image being cropped.
            Default: 0.5
    Returns:
        Tensor, Tensor: cropped images
    """
    T_length = sequence1.size(dim=0)
    transformed1 = sequence1.clone()
    transformed2 = sequence2.clone()
    if random.random() > p:
        transformed1 = torch.transpose(sequence1, 2, 3)
        transformed2 = torch.transpose(sequence2, 2, 3)
    return transformed1, transformed2


def make_dataset(config):
    cfg = copy.deepcopy(config)
    type = cfg.pop("type")
    if type == "REDSDataset":
        from .REDSDataset import REDSDataset

        return REDSDataset(**cfg)


def make_dataloader(config):
    cfg = copy.deepcopy(config)
    cfg["dataset"] = make_dataset(cfg["dataset"])
    return DataLoader(**cfg)
