import copy
import math
from torch.utils.data import DataLoader
from .REDSDataset import REDSDataset
from .utils import generate_segment_indices


def make_dataset(config):
    cfg = copy.deepcopy(config)
    type = cfg.pop("type")
    if type == "REDSDataset":
        return REDSDataset(**cfg)
    else:
        raise NotImplementedError(f"Dataset {type} is not supported yet.")


def make_dataloader(dataset, config):
    cfg = copy.deepcopy(config)
    cfg["dataset"] = dataset
    return DataLoader(**cfg)


def make_train_val_dataloader(config):
    cfg = copy.deepcopy(config)

    train_set = make_dataset(config["train_dataloader"]["dataset"])
    train_loader = make_dataloader(train_set, config["train_dataloader"])
    val_set = make_dataset(config["val_dataloader"]["dataset"])
    val_loader = make_dataloader(val_set, config["val_dataloader"])

    num_iter_per_epoch = math.ceil(
        len(train_set) / (cfg["train_dataloader"]["batch_size"])
    )
    total_iters = int(cfg["train"]["total_iter"])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

    return train_loader, val_loader, total_epochs, total_iters
