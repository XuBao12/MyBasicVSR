import argparse
import os
import json
import torch
import math
import logging
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import make_model
from dataset import make_train_val_dataloader
from utils import (
    load_config,
    check_file,
    copy_cfg,
    get_time_str,
    get_root_logger,
    get_env_info,
    log_infos,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an editor")

    parser.add_argument("--config", default="../config/train.yml")
    args = parser.parse_args()
    return args


def train_pipeline(args):
    # load config
    check_file(args.config)
    config = load_config(args.config, is_train=True)
    print("config is loaded from command line")
    copy_cfg(config)

    log_file = os.path.join(
        config["path"]["log"], f"train_{config['exp_name']}_{get_time_str()}.log"
    )
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())

    tb_logger = SummaryWriter(log_dir=config["path"]["log"])

    train_loader, val_loader, total_epochs, total_iters = make_train_val_dataloader(
        config
    )

    model = make_model(config)

    # TODO resume training
    start_epoch = 0
    current_iter = 0

    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
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

            # log
            if current_iter % config["logger"]["print_freq"] == 0:
                log_infos(
                    config["logger"],
                    epoch,
                    current_iter,
                    model.get_current_learning_rate(),
                    model.loss_dict,
                    tb_logger,
                )

            # save models and training states
            if current_iter % config["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                print("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if current_iter % config["val"]["val_freq"] == 0:
                logger.info(f"Validate model at iteration:{current_iter}.")
                model.validation(val_loader, current_iter, config["val"]["save_img"])

            train_data = next(iter(train_loader))
        # end of iter

    # end of epoch
    logger.info("End training.")
    model.save(epoch=-1, current_iter=-1)

    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args)
