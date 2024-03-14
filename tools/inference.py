import os
import argparse
import json
import torch
from torchvision.utils import save_image

from model import BasicVSR
from dataset import generate_segment_indices
from utils import resize_sequences, evaluate_no_avg


def parse_args():
    parser = argparse.ArgumentParser(description="Inference BasicVSR model")

    parser.add_argument(
        "--gt_dir", default="/root/autodl-tmp/MyBasicVSR/data/train_sharp/000"
    )
    parser.add_argument(
        "--lq_dir", default="/root/autodl-tmp/MyBasicVSR/data/train_sharp_BI_wa/X4/000"
    )
    parser.add_argument("--log_dir", default="../work_dir/BI/test/000")
    parser.add_argument(
        "--spynet_pretrained", default="../checkpoint/spynet_20210409-c6c1bd09.pth"
    )
    parser.add_argument(
        "--checkpoint", default="../checkpoint/basicvsr_reds4_pretrained.pth"
    )
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--from_video", default=False)
    parser.add_argument("--save_image", default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    save_path = args.log_dir
    os.makedirs(f"{save_path}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    model = BasicVSR(
        scale_factor=args.scale_factor, spynet_pretrained=args.spynet_pretrained
    )
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)

    if args.from_video:
        # TODO: 完成此部分
        gt_sequences, lq_sequences = None, None
    else:
        gt_sequences, lq_sequences = generate_segment_indices(
            args.gt_dir, args.lq_dir, num_input_frames=100
        )
    gt_sequences = gt_sequences.unsqueeze(0).to(device)
    lq_sequences = lq_sequences.unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        metrics = {"psnr": {}, "ssim": {}}
        pred_sequences = model(lq_sequences)
        bicubic_sequences = resize_sequences(
            lq_sequences, (gt_sequences.size(dim=3), gt_sequences.size(dim=4))
        )

        eval_result = evaluate_no_avg(pred_sequences, gt_sequences, metrics)
        bi_eval_result = evaluate_no_avg(bicubic_sequences, gt_sequences, metrics)

        if args.save_image:
            os.makedirs(f"{save_path}/images/SR", exist_ok=True)
            os.makedirs(f"{save_path}/images/BICUBIC", exist_ok=True)
            for i in range(pred_sequences.size(1)):
                save_image(
                    pred_sequences[0][i], f"{save_path}/images/SR/{i:08d}.png", nrow=5
                )
                save_image(
                    bicubic_sequences[0][i],
                    f"{save_path}/images/BICUBIC/{i:08d}.png",
                    nrow=5,
                )

        inference_rst_name = f"{save_path}/inference_rst.json"
        bicubic_rst_name = f"{save_path}/bicubic_rst.json"
        with open(inference_rst_name, "w") as file_obj:
            json.dump(eval_result, file_obj)
        with open(bicubic_rst_name, "w") as file_obj:
            json.dump(bi_eval_result, file_obj)
