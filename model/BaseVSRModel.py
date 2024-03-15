from .BaseSRModel import BaseSRModel


class BaseVSRModel(BaseSRModel):
    def __init__(self, config):
        super(BaseVSRModel, self).__init__(config)
        if self.is_train:
            self.fix_flow_iter = self.cfg["train"].get("fix_flow")

    def setup_optimizers(self):
        train_opt = self.cfg["train"]
        flow_lr_mul = train_opt.get("flow_lr_mul", 1)
        if flow_lr_mul == 1:
            optim_params = self.backbone.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.backbone.named_parameters():
                if "spynet" in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    "params": normal_params,
                    "lr": train_opt["optimizer"]["lr"],
                },
                {"params": flow_params, "lr": train_opt["optimizer"]["lr"] * flow_lr_mul},
            ]

        optim_type = train_opt["optimizer"].pop("type")
        self.optimizer = self.get_optimizer(
            optim_type, optim_params, **train_opt["optimizer"]
        )
        self.optimizers.append(self.optimizer)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            if current_iter == 1:
                for name, param in self.backbone.named_parameters():
                    if "spynet" in name or "edvr" in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                self.backbone.requires_grad_(True)

        super(BaseVSRModel, self).optimize_parameters(current_iter)
