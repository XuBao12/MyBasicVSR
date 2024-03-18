import logging

initialized_logger = {}


def get_root_logger(logger_name="basicsr", log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = "%(asctime)s %(levelname)s: %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    if log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def log_infos(config, epoch, current_iter, lrs, loss_dict, tb_logger):
    logger = get_root_logger()
    message = f"[epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:("
    for v in lrs:
        message += f"{v:.3e},"
    message += ")] "

    # TODO 统计时间
    # time and estimated time
    # if "time" in log_vars.keys():
    #     iter_time = log_vars.pop("time")
    #     data_time = log_vars.pop("data_time")

    #     total_time = time.time() - start_time
    #     time_sec_avg = total_time / (current_iter - start_iter + 1)
    #     eta_sec = time_sec_avg * (max_iters - current_iter - 1)
    #     eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
    #     message += f"[eta: {eta_str}, "
    #     message += f"time (data): {iter_time:.3f} ({data_time:.3f})] "

    # other items, especially losses
    for k, v in loss_dict.items():
        message += f"{k}: {v:.4e} "
        # tensorboard logger
        if config["use_tb_logger"]:
            tb_logger.add_scalar(f"losses/{k}", v, current_iter)
    logger.info(message)


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    msg = (
        "Version Information: "
        f"\n\tPyTorch: {torch.__version__}"
        f"\n\tTorchVision: {torchvision.__version__}"
    )
    return msg
