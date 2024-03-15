import yaml
import torch


def load_config(config_file, is_train=True):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["is_train"] = is_train
    if config["num_gpu"] == "auto":
        config["num_gpu"] = torch.cuda.device_count()

    # _merge(defaults, config)
    return config
