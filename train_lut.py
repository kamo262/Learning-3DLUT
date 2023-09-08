import argparse
import json
import os
import math
import time
import datetime
import sys
from pathlib import Path

import torch

from models_x import *
from datasets import *
from color import *
from transforms import *


def adjust_learning_rate(optimizer, epoch, opt):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch <= opt.warmup_epochs:
        lr = opt.lr * epoch / opt.warmup_epochs
    else:
        lr = (
            opt.lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - opt.warmup_epochs)
                    / (opt.n_epochs - opt.warmup_epochs)
                )
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lut_dim", type=int, default=33, help="lut dim")
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="epoch to start training from, 1 starts from scratch, >1 starts from saved checkpoints",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=20, help="total number of epochs of training"
    )
    parser.add_argument("--dataset_path", type=Path, help="path of the dataset")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="path to save model",
    )
    parser.add_argument(
        "--loss_type", type=str, default="cie2000", help="mse | cie2000"
    )
    parser.add_argument(
        "--transforms", type=str, default="none", help="(none | resize | resizedcrop)"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="size of input image. when transforms=none, this value is ignored.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument(
        "--warmup_epochs", type=int, default=2, help="epochs to warmup LR"
    )
    parser.add_argument(
        "--b1",
        type=float,
        default=0.9,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--lambda_smooth", type=float, default=100.0, help="smooth regularization"
    )
    parser.add_argument(
        "--smooth_type",
        type=str,
        default="modified",
        help="type of smooth regularization",
    )
    parser.add_argument(
        "--lambda_monotonicity",
        type=float,
        default=100.0,
        help="monotonicity regularization",
    )
    parser.add_argument(
        "--monotonicity_margin", type=float, default=0.012, help="monotonicity margin"
    )
    parser.add_argument(
        "--lambda_identity",
        type=float,
        default=0.001,
        help="identity regularization",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=0,
        help="number of cpu threads to use during batch generation",
    )
    opt = parser.parse_args()
    os.makedirs(opt.output_dir, exist_ok=True)

    print(opt)
    with open(opt.output_dir / "opts.json", "w") as f:
        json.dump(vars(opt), f, default=lambda x: str(x))

    # Tensor type
    Tensor = torch.FloatTensor

    # Loss functions
    assert opt.loss_type in ["mse", "cie2000"]
    if opt.loss_type == "mse":
        criterion_pixelwise = torch.nn.MSELoss()
    elif opt.loss_type == "cie2000":
        criterion_pixelwise = cie2000_loss_image

    LUT = LUT_3D(dim=opt.lut_dim)
    TV3 = TV_3D(
        dim=opt.lut_dim, mn_margin=opt.monotonicity_margin, tv_type=opt.smooth_type
    )

    # Optimizers

    optimizer = torch.optim.AdamW(
        LUT.parameters(),
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
    )

    if opt.epoch != 1:
        # Load pretrained models
        print(f"load@epoch{opt.epoch}")
        checkpoint = torch.load(opt.output_dir / f"LUT_{opt.epoch}.pth")
        LUT.load_state_dict(checkpoint["LUT"])
        optimizer.load_state_dict(checkpoint["optim"])

    if opt.transforms == "none":
        transforms = None
    elif opt.transforms == "resize":
        transforms = Resize(opt.input_size)
    elif opt.transforms == "resizedcrop":
        transforms = RandomResizedCrop(opt.input_size, scale=(0.5, 1.0))

    dataset = Dataset(opt.dataset_path, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
    )

    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    min_mse = 1e9
    max_epoch = 0
    for epoch in range(opt.epoch, opt.n_epochs + 1):
        mse_avg = 0.0
        for i, batch in enumerate(dataloader):
            n_iters = i

            adjust_learning_rate(optimizer, i / len(dataloader) + epoch, opt)

            # Model inputs
            img_raw = batch["raw"]
            img_jpeg = batch["jpeg"]

            optimizer.zero_grad()

            img_lut = LUT(img_raw)

            mse = criterion_pixelwise(img_lut, img_jpeg)

            # Pixel-wise loss
            tv, mn, identity = TV3(LUT)
            loss = (
                mse
                + opt.lambda_smooth * tv
                + opt.lambda_monotonicity * mn
                + opt.lambda_identity * identity
            )
            mse_avg += mse.item()

            loss.backward()
            optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            # Print log
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [mse: %f, tv: %f, mn: %f, id: %f, loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i + 1,
                    len(dataloader),
                    mse.item(),
                    tv,
                    mn,
                    identity,
                    loss.item(),
                )
            )

        avg_mse = mse_avg / len(dataloader)
        mse_avg = 0.0
        if avg_mse < min_mse:
            min_mse = avg_mse
            max_epoch = epoch

        # Determine approximate time left
        epochs_left = opt.n_epochs - epoch
        time_left = datetime.timedelta(seconds=epochs_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            " [MSE: %f, tv: %f, mn: %f, id: %f] [min MSE: %f, epoch: %d] ETA: %s\n"
            % (avg_mse, tv, mn, identity, min_mse, max_epoch, time_left)
        )

        # Save model checkpoints
        torch.save(
            {"LUT": LUT.state_dict(), "optim": optimizer.state_dict()},
            opt.output_dir / f"LUT_{epoch}.pth",
        )
        with open(opt.output_dir / "result.txt", "a") as file:
            file.write(
                " [MSE: %f, tv: %f, mn: %f, id: %f] [min MSE: %f, epoch: %d]\n"
                % (avg_mse, tv, mn, identity, min_mse, max_epoch)
            )
