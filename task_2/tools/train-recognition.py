import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from torch import optim
from torch.nn.functional import ctc_loss, log_softmax
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_lr_finder import LRFinder

from contest.common import get_logger
from contest.recognition.dataset import RecognitionDataset
from contest.recognition.model import get_model
from contest.recognition.transforms import get_train_transforms
from contest.recognition.utils import abc


DEFAULT_INPUT_SIZE = '640x128'
# DEFAULT_INPUT_SIZE = "320x64"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("--epochs", "-e", dest="epochs", type=int, help="number of train epochs", default=16)
    parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, help="batch size", default=128)
    parser.add_argument("--lr", "-lr", dest="lr", type=float, help="lr", default=3e-4)
    parser.add_argument("--input_wh", "-wh", dest="input_wh", type=str, help="model input size", default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--load", "-l", dest="load", type=str, help="pretrained weights", default=None)
    parser.add_argument("-o", "--output_dir", dest="output_dir", default="runs/recognition_baseline",
                        help="dir to save log and models")
    return parser.parse_args()


def train(model, criterion, optimizer, lr_scheduler, train_dataloader, logger, device):
    # TODO TIP: There's always a chance to overfit to training data...
    model.train()
    epoch_losses = []
    tqdm_iter = tqdm.tqdm(train_dataloader)

    for i, batch in enumerate(tqdm_iter):
        images = batch["images"].to(device)
        seqs = batch["seqs"]
        seq_lens = batch["seq_lens"]

        # TODO TIP: What happens here is explained in seminar 06.
        seqs_pred = model(images).cpu()
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = criterion(log_probs, seqs, seq_lens_pred, seq_lens)
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    logger.info("Epoch finished! Loss: {:.5f}".format(np.mean(epoch_losses)))
    return np.mean(epoch_losses)


def validate(model, criterion, val_dataloader, logger, device):
    model.eval()
    epoch_losses = []
    tqdm_iter = tqdm.tqdm(val_dataloader)

    for i, batch in enumerate(tqdm_iter):
        images = batch["images"].to(device)
        seqs = batch["seqs"]
        seq_lens = batch["seq_lens"]

        seqs_pred = model(images).cpu()
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

        loss = criterion(log_probs, seqs, seq_lens_pred, seq_lens)
        epoch_losses.append(loss.item())
        tqdm_iter.set_description(f"mean loss: {np.mean(epoch_losses):.4f}")

    logger.info("Epoch finished! Loss: {:.5f}".format(np.mean(epoch_losses)))
    return np.mean(epoch_losses)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = get_logger(os.path.join(args.output_dir, "train.log"))
    logger.info("Start training with params:")
    image_w, image_h = list(map(int, args.input_wh.split('x')))

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    logger.info(f"Device: {device}")

    # TODO TIP: The provided model has _lots_ of params to tune.
    # TODO TIP: Also, it's architecture is not the only option.
    # Is recurrent part necessary at all?
    model = get_model(image_w, image_h)

    if args.load is not None:
        with open(args.load, "rb") as fp:
            state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(state_dict)

    model.to(device)
    logger.info(f"Model type: {model.__class__.__name__}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # TODO TIP: Ctc_loss is not the only choice here.
    criterion = ctc_loss

    # TODO TIP: Think of data labels. What kinds of imbalance may it bring?
    # You may benefit from learning about samplers (at torch.utils.data).
    # TODO TIP: And of course given lack of data you should augment it.
    train_transforms = get_train_transforms((image_w, image_h))
    train_dataset = RecognitionDataset(
        args.data_path,
        os.path.join(args.data_path, "train_recognition.json"),
        abc=abc,
        transforms=train_transforms,
        split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.collate_fn
    )
    val_dataset = RecognitionDataset(
        args.data_path,
        os.path.join(args.data_path, "train_recognition.json"),
        abc=abc,
        transforms=train_transforms,
        split="val"
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        collate_fn=train_dataset.collate_fn
    )
    best_model_info = {"epoch": -1, "train_loss": np.inf, "val_loss": np.inf}

    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_dataloader, end_lr=10, num_iter=100)
    lr_finder.plot()
    lr_finder.reset()
    plt.savefig("LRvsLoss.png")
    plt.close()

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_dataloader)
    )

    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1} / {args.epochs}.")
        train_loss = train(model, criterion, optimizer, lr_scheduler, train_dataloader, logger, device)
        val_loss = validate(model, criterion, val_dataloader, logger, device)

        if val_loss < best_model_info["val_loss"]:
            with open(os.path.join(args.output_dir, "CP-best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)
            best_model_info["epoch"] = epoch
            best_model_info["train_loss"] = train_loss
            best_model_info["val_loss"] = val_loss
            logger.info(f"Train loss: {train_loss:.5f}\tval loss: {val_loss:.5f} (best)")
        else:
            best_val = best_model_info["val_loss"]
            best_epoch = best_model_info["epoch"]
            logger.info(f"Train loss: {train_loss:.5f}\tval loss: {val_loss:.5f} (best {best_val:.5f} on {best_epoch})")

    logger.info("Train finished. Best val loss: {}, best epoch: {}".format(best_model_info["val_loss"], best_model_info["epoch"]))

    with open(os.path.join(args.output_dir, "CP-last.pth"), "wb") as fp:
        torch.save(model.state_dict(), fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
