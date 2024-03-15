import copy
from torch.utils.data import DataLoader
from .REDSDataset import REDSDataset
from .utils import generate_segment_indices


def make_dataset(config):
    cfg = copy.deepcopy(config)
    type = cfg.pop("type")
    if type == "REDSDataset":
        return REDSDataset(**cfg)


def make_dataloader(config):
    cfg = copy.deepcopy(config)
    cfg["dataset"] = make_dataset(cfg["dataset"])
    return DataLoader(**cfg)
