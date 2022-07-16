import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from contest.common import get_logger
from contest.segmentation.dataset import DetectionDataset
from contest.segmentation.models import get_model
from contest.segmentation.transforms import get_train_transforms


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("-e", "--epochs", dest="epochs", default=8, type=int, help="number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-s", "--image_size", dest="image_size", default=256, type=int, help="input image size")
    parser.add_argument("-lr", "--learning_rate", dest="lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("-l", "--load", dest="load", default=None, help="load file model")
    parser.add_argument("-o", "--output_dir", dest="output_dir", default="runs/segmentation_baseline",
                        help="dir to save log and models")
    return parser.parse_args()


def train(model, optimizer, lr_scheduler, criterion, train_dataloader, logger, device=None):
    model.train()
    epoch_losses = []
    tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    accum_batch_size = 128

    for i, batch in tqdm_iter:
        imgs, true_masks = batch["image"], batch["mask"].float()
        masks_pred = model(imgs.to(device)).float()
        masks_probs = torch.sigmoid(masks_pred).to(device)

        accum_iter = accum_batch_size / len(imgs)

        loss = criterion(masks_probs.view(-1), true_masks.view(-1).to(device)).cpu()
        loss = loss / accum_iter
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

        loss.backward()

        if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

    logger.info(f"Epoch finished! Loss: {np.mean(epoch_losses):.5f}")

    return np.mean(epoch_losses)


def validate(model, criterion, val_dataloader, logger, device=None):
    model.eval()
    epoch_losses = []
    tqdm_iter = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    for i, batch in tqdm_iter:
        imgs, true_masks = batch["image"], batch["mask"].float()
        masks_pred = model(imgs.to(device)).float()
        masks_probs = torch.sigmoid(masks_pred).to(device)

        loss = criterion(masks_probs.view(-1), true_masks.view(-1).to(device)).cpu()
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

    logger.info(f"Epoch finished! Test loss: {np.mean(epoch_losses):.5f}")

    return np.mean(epoch_losses)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = get_logger(os.path.join(args.output_dir, "train.log"))
    logger.info("Start training with params:")

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    logger.info(f"Device: {device}")
    model = get_model()

    if args.load is not None:
        with open(args.load, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)
    logger.info(f"Model type: {model.__class__.__name__}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    # criterion = dice_loss

    train_transforms = get_train_transforms(args.image_size)
    train_dataset = DetectionDataset(
        args.data_path,
        os.path.join(args.data_path, "train_segmentation.json"),
        transforms=train_transforms,
        split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True, shuffle=True, drop_last=True
    )
    val_dataset = DetectionDataset(
        args.data_path,
        os.path.join(args.data_path, "train_segmentation.json"),
        transforms=train_transforms,
        split="val"
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4,
        num_workers=4,
        pin_memory=True, shuffle=True, drop_last=True
    )
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # lr_finder.range_test(train_dataloader, end_lr=10, num_iter=100)
    # lr_finder.plot()
    # lr_finder.reset()
    plt.savefig("LRvsLoss.png")
    plt.close()

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dataloader)
    )
    logger.info(f"Length of train = {len(train_dataset)}")
    best_model_info = {"epoch": -1, "train_loss": np.inf, "val_loss": np.inf}

    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}.")

        train_loss = train(model, optimizer, lr_scheduler, criterion, train_dataloader, logger, device)
        val_loss = validate(model, criterion, val_dataloader, logger, device)

        if val_loss < best_model_info["val_loss"]:
            with open(os.path.join(args.output_dir, "CP-best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)
            best_model_info["epoch"] = epoch
            best_model_info["train_loss"] = train_loss
            best_model_info["val_loss"] = val_loss
            logger.info(f"Train loss: {train_loss:.5f}, val loss: {val_loss:.5f} (best)")
        else:
            logger.info(f"Train loss: {train_loss:.5f}, val loss: {val_loss:.5f} (best {best_model_info['val_loss']:.5f})")

    logger.info("Train finished. Best val loss: {}, best epoch: {}".format(best_model_info["val_loss"], best_model_info["epoch"]))

    with open(os.path.join(args.output_dir, "CP-last.pth"), "wb") as fp:
        torch.save(model.state_dict(), fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
