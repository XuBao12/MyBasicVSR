import yaml
import torch
import os
import os.path as osp
import glob


def load_config(args, is_train=True):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["is_train"] = is_train
    if config["num_gpu"] == "auto":
        config["num_gpu"] = torch.cuda.device_count()

    if config.get("path") is None:
        config["path"] = {}
    exp_root = os.path.join(config["work_dir"], config["name"])
    config["path"]["exp_root"] = exp_root
    config["path"]["log"] = exp_root
    config["path"]["models"] = osp.join(exp_root, "models")
    config["path"]["training_states"] = osp.join(exp_root, "training_states")
    config["path"]["visualization"] = osp.join(exp_root, "visualization")
    for dir in config["path"].values():
        os.makedirs(dir, exist_ok=True)

    config["auto_resume"] = args.auto_resume
    if config["auto_resume"]:
        state_path = config["path"]["training_states"]
        state_files = glob.glob(os.path.join(state_path, "*.state"), recursive=True)
        resume_state_path = max(
            state_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        )
        if resume_state_path:
            config["path"]["resume_state"] = resume_state_path
            resume_state = torch.load(resume_state_path)
            config["model"]["pretrained"] = osp.join(
                config["path"]["models"], f"backbone_{resume_state['iter']}.pth"
            )
        else:
            resume_state = None
    else:
        resume_state = None
    copy_cfg(config)

    return config, resume_state


def copy_cfg(config):
    filename = osp.join(config["path"]["exp_root"], "config.yml")
    with open(filename, "w") as file:
        yaml.safe_dump(config, file)
