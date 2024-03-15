import copy
from .CharbonnierLoss import CharbonnierLoss


def make_loss(config):
    cfg = copy.deepcopy(config)
    type = cfg.pop("type")
    if type == "CharbonnierLoss":
        return CharbonnierLoss(**cfg)
    else:
        raise NotImplementedError(f"Loss {type} is not supported yet.")
