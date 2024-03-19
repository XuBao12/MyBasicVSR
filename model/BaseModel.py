import os
import torch
import copy
from torch import nn
from copy import deepcopy
from utils import get_root_logger


class BaseModel(nn.Module):
    """Base model."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.cfg = copy.deepcopy(config)
        self.device = torch.device("cuda" if self.cfg["num_gpu"] != 0 else "cpu")
        self.is_train = self.cfg["is_train"]
        self.schedulers = []
        self.optimizers = []
        self.loss_dict = {}

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def save_network(self, net, net_label, current_iter, param_key="params"):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = "latest"
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = os.path.join(self.cfg["path"]["models"], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            # net_ = self.get_bare_model(net_) 并行计算
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith("module."):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict
        # 保存的pth文件：{"params":{state_dict}，"params_ema":{state_dict}}，根据有无mea决定是否包含第二项，"params"以及"params_ema"统称为"param_key"
        torch.save(save_dict, save_path)

    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
            }
            for o in self.optimizers:
                state["optimizers"].append(o.state_dict())
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())
            save_filename = f"{current_iter}.state"
            save_path = os.path.join(self.cfg["path"]["training_states"], save_filename)
            torch.save(state, save_path)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == "Adam":
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == "AdamW":
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == "Adamax":
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == "SGD":
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == "ASGD":
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == "RMSprop":
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == "Rprop":
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supported yet.")
        return optimizer

    def setup_schedulers(self):
        """Set up schedulers."""
        train_cfg = self.cfg["train"]
        scheduler_type = train_cfg["scheduler"].pop("type")
        if scheduler_type == "CosineAnnealingLR":
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, **train_cfg["scheduler"]
                    )
                )
        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_type} is not implemented yet."
            )

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def load_network(self, net, load_path, strict=True, param_key="params"):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        load_net = torch.load(load_path)
        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
                logger.info("Loading: params_ema does not exist, use params.")
            load_net = load_net[param_key]
        logger.info(
            f"Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}]."
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v["initial_lr"] for v in optimizer.param_groups])
        return init_lr_groups_l

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group["lr"] = lr

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if (
            hasattr(self, "best_metric_results")
            and dataset_name in self.best_metric_results
        ):
            return
        elif not hasattr(self, "best_metric_results"):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.cfg["val"]["metrics"].items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def model_ema(self, decay=0.999):
        backbone = self.backbone

        backbone_params = dict(backbone.named_parameters())
        backbone_ema_params = dict(self.backbone_ema.named_parameters())

        for k in backbone_ema_params.keys():
            backbone_ema_params[k].data.mul_(decay).add_(
                backbone_params[k].data, alpha=1 - decay
            )
