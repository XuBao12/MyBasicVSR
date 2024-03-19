import argparse
import os
import logging
from torch.utils.tensorboard import SummaryWriter

from model import make_model
from dataset import make_train_val_dataloader
from utils import (
    load_config,
    check_file,
    get_time_str,
    get_root_logger,
    get_env_info,
    log_infos,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an editor")

    parser.add_argument("config")
    parser.add_argument("--auto_resume", action="store_true")
    args = parser.parse_args()
    return args


def train_pipeline(args):
    # load config
    check_file(args.config)
    config, resume_state = load_config(args, is_train=True)

    # logger
    log_file = os.path.join(
        config["path"]["log"], f"train_{config['name']}_{get_time_str()}.log"
    )
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info("config is loaded from command line")
    logger.info(get_env_info())
    tb_logger = SummaryWriter(log_dir=config["path"]["log"])

    # dataloader and model
    train_loader, val_loader, total_epochs, total_iters = make_train_val_dataloader(
        config
    )
    model = make_model(config)

    # resume training
    if resume_state:
        model.resume_training(resume_state)  # handle optimizers and schedulers
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
        logger.info(
            f"Resuming training from epoch: {start_epoch}, iter: {current_iter}."
        )
    else:
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
