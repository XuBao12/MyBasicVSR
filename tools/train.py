import argparse
import os
import json
import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import make_model
from dataset import make_dataloader
from loss import CharbonnierLoss
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train an editor")

    parser.add_argument("--config", default="../config/train.yml")
    parser.add_argument(
        # "--checkpoint", default="../checkpoint/basicvsr_reds4_pretrained.pth"
        "--checkpoint",
        default=None,
    )
    parser.add_argument("--rst_file", default=None)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--val_interval", default=2, type=int)

    args = parser.parse_args()
    return args


def main(args):
    # load config
    try:
        config_path = os.path.join(log_dir, "config.yaml")
        check_file(config_path)
        config = load_config(config_path)
        print("config is loaded from checkpoint folder")
        config["_resume"] = True
    except:
        check_file(args.config)
        config = load_config(args.config)
        print("config is loaded from command line")

    log_dir = os.path.join(config["work_dir"], config["exp_name"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/models", exist_ok=True)
    os.makedirs(f"{log_dir}/images", exist_ok=True)

    # TODO 更改config一些内容然后记录为yaml存到log_dir下

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(config["train_dataloader"])
    val_loader = make_dataloader(config["val_dataloader"])

    model = make_model(config["model"])
    # TODO resume
    if args.checkpoint:
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
        rst_file_name = f"{log_dir}/rst.json"
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
            os.makedirs(f"{log_dir}/images/epoch{epoch:05}", exist_ok=True)
            val_rst = val_epoch(model, val_loader, epoch, device, log_dir)
            val_results.append(val_rst)

    draw_loss(train_loss, args)
    draw_valrst(val_results, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
