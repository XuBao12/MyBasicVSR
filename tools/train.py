import argparse
import os
import json
import torch
import math
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import make_model
from dataset import make_train_val_dataloader
from utils import load_config, check_file, copy_cfg


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


def train_pipeline(args):
    # load config
    check_file(args.config)
    config = load_config(args.config, is_train=True)
    print("config is loaded from command line")
    copy_cfg(config, config["path"]["exp_root"])

    train_loader, val_loader, total_epochs, total_iters = make_train_val_dataloader(
        config
    )
    print(total_epochs, total_iters)

    model = make_model(config)

    # TODO resume training
    start_epoch = 0
    current_iter = 0

    for epoch in range(start_epoch, total_epochs + 1):
        train_data = next(iter(train_loader))

        while train_data is not None:
            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=config["train"].get("warmup_iter", -1)
            )
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            # TODO log
            print(f"iter:{current_iter}")

            # save models and training states
            if current_iter % config["logger"]["save_checkpoint_freq"] == 0:
                print("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if current_iter % config["val"]["val_freq"] == 0:
                model.validation(val_loader, current_iter, config["val"]["save_img"])

            train_data = next(iter(train_loader))
        # end of iter

    # end of epoch


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
