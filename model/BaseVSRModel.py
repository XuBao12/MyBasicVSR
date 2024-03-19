from .BaseSRModel import BaseSRModel
import torch
from tqdm import tqdm
from utils import tensor2img, calculate_metric, get_root_logger


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
                {
                    "params": flow_params,
                    "lr": train_opt["optimizer"]["lr"] * flow_lr_mul,
                },
            ]

        optim_type = train_opt["optimizer"].pop("type")
        self.optimizer = self.get_optimizer(
            optim_type, optim_params, **train_opt["optimizer"]
        )
        self.optimizers.append(self.optimizer)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(
                    f"Fix flow network and feature extractor for {self.fix_flow_iter} iters."
                )
                for name, param in self.backbone.named_parameters():
                    if "spynet" in name or "edvr" in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                self.backbone.requires_grad_(True)

        super(BaseVSRModel, self).optimize_parameters(current_iter)

    def validation(self, dataloader, current_iter, save_img):
        dataset_cfg = self.cfg["val_dataloader"]["dataset"]
        dataset_name = dataset_cfg["type"]
        with_metrics = self.cfg["val"]["metrics"] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {}
                # num_frame_each_folder = Counter(dataset.data_info["folder"])
                num_frame_each_folder = {
                    i: dataset_cfg["num_input_frames"] for i in range(len(dataloader))
                }
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame,
                        len(self.cfg["val"]["metrics"]),
                        dtype=torch.float32,
                        device="cuda",
                    )
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataloader)
        pbar = tqdm(total=num_folders, unit="folder")
        for i, val_data in enumerate(dataloader):
            # folder = val_data["folder"]
            folder = i
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()

            # evaluate
            if i < num_folders:
                for idx in range(visuals["result"].size(1)):
                    result = visuals["result"][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data["img1"] = result_img
                    if "gt" in visuals:
                        gt = visuals["gt"][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data["img2"] = gt_img

                    if save_img:
                        raise NotImplementedError("saving image is not supported now.")
                        # BUG 主要是数据集那里丢了folder信息，可能要重写数据集代码
                        # img_path = os.path.join(
                        #     self.cfg["path"]["visualization"],
                        #     dataset_name,
                        #     folder,
                        #     f"{idx:08d}_{self.cfg['name']}.png",
                        # )
                        # # image name only for REDS dataset
                        # cv2.imwrite(result_img, img_path)
                        pass

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, (metric_type, kwargs) in enumerate(
                            self.cfg["val"]["metrics"].items()
                        ):
                            result = calculate_metric(metric_type, metric_data, kwargs)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                pbar.update(1)
                pbar.set_description(f"Folder: {folder}")

        pbar.close()

        if with_metrics:
            self._log_validation_metric_values(current_iter, dataset_name)

    def _log_validation_metric_values(self, current_iter, dataset_name):
        # ----------------- calculate the average values for each folder, and for each metric  ----------------- #
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            folder: torch.mean(tensor, dim=0).cpu()
            for (folder, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        total_avg_results = {metric: 0 for metric in self.cfg["val"]["metrics"].keys()}
        for folder, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[folder][idx].item()
        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)
            # update the best metric result
            self._update_best_metric_result(
                dataset_name, metric, total_avg_results[metric], current_iter
            )

        # ------------------------------------------ log the metric ------------------------------------------ #
        logger = get_root_logger()
        log_str = f"Validation {dataset_name}\n"
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f"\t # {metric}: {value:.4f}"
            for folder, tensor in metric_results_avg.items():
                log_str += f"\t # {folder}: {tensor[metric_idx].item():.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\n\t    Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"
        logger.info(log_str)
