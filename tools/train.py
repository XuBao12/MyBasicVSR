import argparse
import os
import json
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import BasicVSR
from dataset import REDSDataset
from loss import CharbonnierLoss
from utils import train_epoch, val_epoch, draw_loss, draw_valrst


def parse_args():
    parser = argparse.ArgumentParser(description="Train an editor")

    parser.add_argument("--gt_dir", default="../data/train_sharp")
    parser.add_argument("--lq_dir", default="../data/train_sharp_BI_na/X4")
    parser.add_argument("--log_dir", default="../work_dir/BI_na/exp_1")
    parser.add_argument(
        "--spynet_pretrained", default="../checkpoint/spynet_20210409-c6c1bd09.pth"
    )
    parser.add_argument(
        "--checkpoint", default="../checkpoint/basicvsr_reds4_pretrained.pth"
    )
    parser.add_argument("--rst_file", default=None)
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


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(f"{args.log_dir}/models", exist_ok=True)
    os.makedirs(f"{args.log_dir}/images", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = REDSDataset(args, is_test=False)
    val_set = REDSDataset(args, is_test=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, pin_memory=True)

    model = BasicVSR(spynet_pretrained=args.spynet_pretrained)
    model.load_state_dict(torch.load(args.checkpoint))

    # 定义损失函数和优化器
    criterion = CharbonnierLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.spynet.parameters(), "lr": 2.5e-5},
            {"params": model.backward_resblocks.parameters()},
            {"params": model.forward_resblocks.parameters()},
            {"params": model.fusion.parameters()},
            {"params": model.upsample1.parameters()},
            {"params": model.upsample2.parameters()},
            {"params": model.conv_hr.parameters()},
            {"params": model.conv_last.parameters()},
        ],
        lr=2e-4,
        betas=(0.9, 0.99),
    )

    max_epoch = args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-7)

    # 保存训练的loss和评价的results(psnr和ssim)
    if args.rst_file:
        rst_file_name = args.rst_file
        with open(rst_file_name, "r") as file_obj:
            load_dict = json.load(file_obj)
        train_loss = load_dict["train_loss"]  # a list
        val_results = load_dict["val_results"]  # a list
    else:
        rst_file_name = f"{args.log_dir}/rst.json"
        train_loss = []  # 记录每次epoch的平均loss, len(train_loss) = len(max_epoch)
        val_results = []  # list of dict

    for epoch in range(max_epoch):
        # train
        train_epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            train_loader,
            epoch,
            device,
            train_loss,
        )
        with open(rst_file_name, "w") as file_obj:
            json.dump(
                {"epoch": epoch, "train_loss": train_loss, "val_results": val_results},
                file_obj,
            )

        # val
        if (epoch + 1) % args.val_interval == 0:
            os.makedirs(f"{args.log_dir}/images/epoch{epoch:05}", exist_ok=True)
            val_rst = val_epoch(model, val_loader, epoch, device, args.log_dir)
            val_results.append(val_rst)

    draw_loss(train_loss, args)
    draw_valrst(val_results, args)
