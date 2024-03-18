from .utils import resize_sequences, check_file
from .train import train_epoch, val_epoch, test_model
from .metrics import evaluate, evaluate_no_avg, cal_psnr, cal_ssim
from .visualization import draw_loss, draw_valrst
from .config import load_config, copy_cfg
from .misc import tensor2img, get_time_str
from .logger import get_root_logger, get_env_info, log_infos
from copy import deepcopy


def calculate_metric(type, data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    if type == "psnr":
        metric = cal_psnr(**data, **opt)
    if type == "ssim":
        metric = cal_ssim(**data, **opt)
    return metric
