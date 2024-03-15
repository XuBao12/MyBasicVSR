import copy
from .BasicVSR import BasicVSR
from .SPyNet import SPyNet


def make_backbone(config):
    cfg = copy.deepcopy(config)
    type = cfg.pop("type")
    if type == "BasicVSR":
        return BasicVSR(**cfg)
    else:
        raise NotImplementedError(f"Backbone {type} is not supported yet.")
