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
from utils import load_config, check_file


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
    try:
        config_path = os.path.join(log_dir, "config.yaml")
        check_file(config_path)
        config = load_config(config_path)
        print("config is loaded from checkpoint folder")
        config["_resume"] = True
    except:
        check_file(args.config)
        config = load_config(args.config, is_train=True)
        print("config is loaded from command line")

    log_dir = os.path.join(config["work_dir"], config["exp_name"])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/models", exist_ok=True)
    os.makedirs(f"{log_dir}/images", exist_ok=True)

    # TODO 更改config一些内容然后记录为yaml存到log_dir下

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(config["train_dataloader"])
    val_loader = make_dataloader(config["val_dataloader"])

    model = make_model(config)

    start_epoch = 0
    current_iter = 0
    # TODO 记录（计算）总的epoch和iter
    total_epochs = 10000
    total_iters = 300000

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

            # TODO save models and training states

            # TODO validation
            train_data = next(iter(train_loader))
        # end of iter

    # end of epoch


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
