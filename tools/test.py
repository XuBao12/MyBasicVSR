import argparse
import os
import json

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import BasicVSR
from dataset import REDSDataset
from loss import CharbonnierLoss
from utils import train_epoch, val_epoch, draw_loss, draw_valrst, test_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train an editor")

    parser.add_argument(
        "--gt_dir", default="/root/autodl-tmp/MyBasicVSR/data/train_sharp"
    )
    parser.add_argument(
        "--lq_dir", default="/root/autodl-tmp/MyBasicVSR/data/train_sharp_bicubic/X4"
    )
    parser.add_argument("--log_dir", default="../work_dir/bicubic/exp_1")
    parser.add_argument(
        "--spynet_pretrained", default="../checkpoint/spynet_20210409-c6c1bd09.pth"
    )
    parser.add_argument(
        "--checkpoint", default="../checkpoint/basicvsr_reds4_pretrained.pth"
    )
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--patch_size", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--num_input_frames", default=5, type=int)
    parser.add_argument("--val_interval", default=2, type=int)
    parser.add_argument("--max_keys", default=270, type=int)
    parser.add_argument("--filename_tmpl", default="{:08d}.png")
    parser.add_argument("--val_partition", default="REDS4")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(f"{args.log_dir}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = REDSDataset(args, is_test=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)

    model = BasicVSR(
        scale_factor=args.scale_factor, spynet_pretrained=args.spynet_pretrained
    )
    model.load_state_dict(torch.load(args.checkpoint))

    test_rst, bicubic_rst = test_model(
        model, test_loader, device, save_path=args.log_dir, save_img=True
    )

    inference_rst_name = f"{args.log_dir}/inference_rst.json"
    bicubic_rst_name = f"{args.log_dir}/bicubic_rst.json"
    with open(inference_rst_name, "w") as file_obj:
        json.dump(test_rst, file_obj)
    with open(bicubic_rst_name, "w") as file_obj:
        json.dump(bicubic_rst, file_obj)


if __name__ == "__main__":
    main()
