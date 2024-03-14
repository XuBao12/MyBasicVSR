from tqdm import tqdm
import os
from utils import resize_sequences
from torchvision.utils import save_image

from .metrics import evaluate

import torch


def train_epoch(
    model, optimizer, criterion, scheduler, train_loader, epoch, device, train_loss
):
    model.to(device)
    model.train()
    train_epoch_loss = (
        []
    )  # 记录epoch里每次迭代的loss值, len(train_epoch_loss) = len(train_loader)

    # fix SPyNet and EDVR at first 5000 iteration
    if epoch < 5000:
        for name, value in model.named_parameters():
            if "spynet" in name or "edvr" in name:
                value.requires_grad_(False)
    elif epoch == 5000:
        # train all the parameters
        model.requires_grad_(True)

    with tqdm(train_loader, ncols=100) as pbar:
        for idx, (gt_sequences, lq_sequences) in enumerate(pbar):
            gt_sequences = gt_sequences.to(device)
            lq_sequences = lq_sequences.to(device)
            pred_sequences = model(lq_sequences)

            optimizer.zero_grad()
            loss = criterion(pred_sequences, gt_sequences)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_epoch_loss.append(loss.item())

            pbar.set_description(f"[Epoch {epoch + 1}]")
            pbar.set_postfix({"loss": f"{loss.data:.3f}"})

    train_loss.append(sum(train_epoch_loss) / len(train_epoch_loss))


def val_epoch(model, val_loader, epoch, device, log_dir):
    metrics = {"psnr": {}, "ssim": {}}
    model.to(device)
    model.eval()

    val_result = {}
    for metric in metrics.keys():
        val_result[metric] = []

    with torch.no_grad():
        for idx, (gt_sequences, lq_sequences) in enumerate(val_loader):
            gt_sequences = gt_sequences.to(device)
            lq_sequences = lq_sequences.to(device)
            pred_sequences = model(lq_sequences)

            # lq插值到gt的size
            # lq_sequences = resize_sequences(lq_sequences, (gt_sequences.size(dim=3), gt_sequences.size(dim=4)))

            eval_result = evaluate(pred_sequences, gt_sequences, metrics)
            for metric in eval_result.keys():
                val_result[metric].append(eval_result[metric])

            # save_image(pred_sequences[0], f'{log_dir}/images/epoch{epoch:05}/{idx}_SR.png', nrow=5)
            # save_image(lq_sequences[0], f'{log_dir}/images/epoch{epoch:05}/{idx}_LQ.png', nrow=5)
            # save_image(gt_sequences[0], f'{log_dir}/images/epoch{epoch:05}/{idx}_GT.png', nrow=5)

        val_avg_rst = {}
        for metric, value in val_result.items():
            avg = sum(value) / len(value)
            val_avg_rst[metric] = avg
            print(f"==[epoch: {epoch} validation]== {metric.upper()}:{avg:.2f}")

        torch.save(model.state_dict(), f"{log_dir}/models/model_{epoch}.pth")
        return val_avg_rst


def test_model(model, test_loader, device, save_path, save_img):
    metrics = {"psnr": {}, "ssim": {}}
    model.to(device)
    model.eval()

    val_result = {}
    bicubic_result = {}
    for metric in metrics.keys():
        val_result[metric] = []
        bicubic_result[metric] = []

    with torch.no_grad():
        with tqdm(test_loader, ncols=100) as pbar:
            for idx, (gt_sequences, lq_sequences) in enumerate(pbar):
                gt_sequences = gt_sequences.to(device)
                lq_sequences = lq_sequences.to(device)
                pred_sequences = model(lq_sequences)

                # lq插值到gt的size
                bicubic_sequences = resize_sequences(
                    lq_sequences, (gt_sequences.size(dim=3), gt_sequences.size(dim=4))
                )

                eval_result = evaluate(pred_sequences, gt_sequences, metrics)
                bi_eval_result = evaluate(bicubic_sequences, gt_sequences, metrics)

                if save_img and idx == 1:
                    os.makedirs(f"{save_path}/images/SR", exist_ok=True)
                    os.makedirs(f"{save_path}/images/BICUBIC", exist_ok=True)
                    os.makedirs(f"{save_path}/images/GT", exist_ok=True)
                    for i in range(pred_sequences.size(1)):
                        save_image(
                            pred_sequences[0][i],
                            f"{save_path}/images/SR/{i}.png",
                            nrow=5,
                        )
                        save_image(
                            bicubic_sequences[0][i],
                            f"{save_path}/images/BICUBIC/{i}.png",
                            nrow=5,
                        )
                        save_image(
                            gt_sequences[0][i], f"{save_path}/images/GT/{i}.png", nrow=5
                        )

                pbar.set_description("Val metrics")
                postfix = {}
                for metric, value in eval_result.items():
                    val_result[metric].append(value)
                    bicubic_result[metric].append(bi_eval_result[metric])
                    postfix[f"{metric.upper()}"] = f"{value:.2f}"
                pbar.set_postfix(postfix)

        for metric, value in val_result.items():
            avg_val = sum(value) / len(value)
            print(f"==[Inference results]== {metric.upper()}:{avg_val:.2f}")

        for metric, value in bicubic_result.items():
            avg_val = sum(value) / len(value)
            print(f"==[Bicubic results]== {metric.upper()}:{avg_val:.2f}")

        return val_result, bicubic_result
