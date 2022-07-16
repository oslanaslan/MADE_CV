"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import NUM_PTS, CROP_SIZE
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys, RandomFlip, RandomImgTransforms
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

# from resnext import resnext101_64x4d, resnext101_64x4d_features

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default='data')
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-2, type=float)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--model", help="Checkpoint name", default=None)
    parser.add_argument("--predict", help="Make prediction without train", action="store_true")

    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):   
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    val_mse_score = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())
        val_mse_score.append(
            ((np.array(pred_landmarks) - np.array(landmarks)) ** 2).reshape(-1)
        )

    val_loss = np.mean(val_loss)
    val_mse_score = np.mean(np.hstack(val_mse_score))

    return val_loss, val_mse_score


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    train_transforms = {
        'train': transforms.Compose([
            RandomFlip(),
            # RandomImgTransforms(),
            ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
            CropCenter(CROP_SIZE),
            TransformByKeys(transforms.ToPILImage(), ("image",)),
            # TransformByKeys(transforms.ColorJitter(brightness=.5, hue=.3), ("image",)),
            # TransformByKeys(transforms.RandomPosterize(bits=2), ("image",)),
            # TransformByKeys(transforms.TrivialAugmentWide(), ("image",)),
            # TransformByKeys(transforms.Grayscale(3), ("image",)),
            TransformByKeys(transforms.ToTensor(), ("image",)),
            TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
        ]),
        'test': transforms.Compose([
            ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
            CropCenter(CROP_SIZE),
            TransformByKeys(transforms.ToPILImage(), ("image",)),
            # TransformByKeys(transforms.Grayscale(3), ("image",)),
            TransformByKeys(transforms.ToTensor(), ("image",)),
            TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
        ])
    }

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms['train'], split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms['test'], split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cuda:0")

    print("Creating model...")
    # model = models.resnext101_32x8d(pretrained=True)
    # model = models.efficientnet_b7(pretrained=True)

    model = models.resnet50(pretrained=True)

    # model.requires_grad_(True)

    # param_count = len(list(model.parameters()))
    # train_last = int(param_count * 0.8)

    # model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2 * NUM_PTS, bias=True)
    # model.classifier[-1].requires_grad_(True)

    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.fc.requires_grad_(True)

    # for idx, param in enumerate(model.parameters()):
    #     if idx > (param_count - train_last):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    # loss_fn = fnn.mse_loss
    loss_fn = fnn.l1_loss
    # scheduler_1 = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.5,
    #     patience=2,
    #     min_lr=1e-8,
    #     threshold=1e-2,
    #     verbose=True
    # )
    # scheduler_2 = optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=10,
    #     gamma=0.5,
    #     verbose=True
    # )
    scheduler_2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular", cycle_momentum=False)
    # scheduler = optim.lr_scheduler.ChainedScheduler([
    #     scheduler_1,
    #     scheduler_2
    # ])

    if args.load:
        if args.model:
            model_path = os.path.join("runs", f"{args.model}_best.pth")
        else:
            model_path = os.path.join("runs", f"{args.name}_best.pth")
        with open(model_path, "rb") as fp:
            best_state_dict = torch.load(fp, map_location="cpu")
            model.load_state_dict(best_state_dict)

    if not args.predict:

        # 2. train & validate
        print("Ready for training...")
        best_val_loss = np.inf

        for epoch in range(args.epochs):
            train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device)
            val_loss, val_mse_score = validate(model, val_dataloader, loss_fn, device=device)
            print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}\tval mse: {:5.2}".format(epoch, train_loss, val_loss, val_mse_score))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                    torch.save(model.state_dict(), fp)
                print('Checkpoint saved')
            
            # scheduler.step(val_loss)
            # scheduler_1.step(val_loss)
            scheduler_2.step()

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), train_transforms['test'], split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
