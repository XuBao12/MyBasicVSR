import torch
from collections import OrderedDict

from backbone import make_backbone
from loss import make_loss
from .BaseModel import BaseModel


class BaseSRModel(BaseModel):
    def __init__(self, config):
        super(BaseSRModel, self).__init__(config)

        # define network
        self.backbone = make_backbone(self.cfg["model"]["backbone"])
        self.backbone.to(self.device)
        # self.print_network(self.backbone)

        # load pretrained models
        load_path = self.cfg["model"].get("pretrained", None)
        if load_path:
            param_key = self.cfg["path"].get("param_key_g", "params")
            self.load_network(
                self.backbone,
                load_path,
                self.cfg["model"].get("strict_load", True),
                param_key,
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.backbone.train()
        train_cfg = self.cfg["train"]

        self.ema_decay = train_cfg.get("ema_decay", 0)
        if self.ema_decay > 0:
            # define network backbone with Exponential Moving Average (EMA)
            # backbone_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.backbone_ema = make_backbone(self.cfg["model"]["backbone"])
            self.backbone_ema.to(self.device)
            # load pretrained model
            load_path = self.cfg["model"].get("pretrained", None)
            if load_path:
                self.load_network(
                    self.backbone_ema,
                    load_path,
                    self.cfg["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy backbone weight
            self.backbone_ema.eval()

        # define losses
        self.criterion = make_loss(train_cfg.get("criterion")).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.cfg["train"]
        optim_params = []
        for k, v in self.backbone.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
        optim_type = train_opt["optimizer"].pop("type")
        self.optimizer = self.get_optimizer(
            optim_type, optim_params, **train_opt["optimizer"]
        )
        self.optimizers.append(self.optimizer)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer.zero_grad()
        self.output = self.backbone(self.lq)
        loss = self.criterion(self.output, self.gt)
        self.loss_dict[self.criterion.__class__.__name__] = loss
        loss.backward()
        self.optimizer.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)  # BUG 未定义model_ema

    def test(self):
        if hasattr(self, "backbone_ema"):
            self.backbone_ema.eval()
            with torch.no_grad():
                self.output = self.backbone_ema(self.lq)
        else:
            self.backbone.eval()
            with torch.no_grad():
                self.output = self.backbone(self.lq)
            self.backbone.train()

    def save(self, epoch, current_iter):
        if hasattr(self, "backbone_ema"):
            self.save_network(
                [self.backbone, self.backbone_ema],
                "backbone",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.backbone, "backbone", current_iter)
        self.save_training_state(epoch, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict
