import yaml
import torch
import os
import os.path as osp


def load_config(config_file, is_train=True):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["is_train"] = is_train
    if config["num_gpu"] == "auto":
        config["num_gpu"] = torch.cuda.device_count()

    if config.get("path") is None:
        config["path"] = {}
    exp_root = os.path.join(config["work_dir"], config["exp_name"])
    config["path"]["exp_root"] = exp_root
    config["path"]["log"] = exp_root
    config["path"]["models"] = osp.join(exp_root, "models")
    config["path"]["training_states"] = osp.join(exp_root, "training_states")
    config["path"]["visualization"] = osp.join(exp_root, "visualization")
    for dir in config["path"].values():
        os.makedirs(dir, exist_ok=True)

    return config


def copy_cfg(config):
    filename = osp.join(config["path"]["exp_root"], "config.yml")
    with open(filename, "w") as file:
        yaml.safe_dump(config, file)
