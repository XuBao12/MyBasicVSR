import copy
from .BasicVSR import BasicVSR
from .SPyNet import SPyNet


def make_model(config):
    cfg = copy.deepcopy(config)
    type = cfg.pop("type")
    if type == "BasicVSR":
        return BasicVSR(**cfg)
