import copy
from .BaseVSRModel import BaseVSRModel


def make_model(config):
    cfg = copy.deepcopy(config)
    type = config["model"]["type"]
    if type == "BaseVSRModel":
        return BaseVSRModel(cfg)
    else:
        raise NotImplementedError(f"Model {type} is not supported yet.")
