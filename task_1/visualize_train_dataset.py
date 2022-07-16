'''
Plot landmarks on train dataset and save to SAVE_PATH
'''
from __future__ import absolute_import
import os
import sys
import gc
import concurrent.futures
from argparse import Namespace, ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import ThousandLandmarksDataset
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import CROP_SIZE


LANDMARKS_PATH = os.path.join("data", "train", "landmarks.csv")
IMAGE_PATH = os.path.join("data", "train", "images")
SAVE_PATH = os.path.join("data", "train_with_landmarks")
CHUNK_SIZE = 15000
BATCH_SIZE = 64


def parse_arguments() -> Namespace:
    parser = ArgumentParser(__doc__)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def draw_landmarks(image: np.array, landmarks: list) -> np.array:
    for point in landmarks:
        x, y = point.astype(np.int)
        cv2.circle(image, (x, y), 1, (128, 0, 128), 1, -1)
    return image


def landmarks_to_points(inp: list) -> list:
    points = zip(inp[0::2], inp[1::2])
    points = map(np.array, points)
    return list(points)


def generate_imgs_with_landmarks(range_: tuple = None) -> None:
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        # TransformByKeys(transforms.Grayscale(3), ("image",)),
        # TransformByKeys(transforms.ToTensor(), ("image",)),
        # TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]), ("image",)),
    ])

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
                                  shuffle=True, drop_last=True)

    for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="training..."):
        images = batch["image"]  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)
        plt.imshow(image)
        plt.savefig(save_path)


    landmarks_df = pd.read_csv(LANDMARKS_PATH, sep='\t')
    # landmarks_df = tmp_df.loc[tmp_df.index[start, end], :].copy()
    start, end = range_ or (0, landmarks_df.shape[0])
    image_names_lst = landmarks_df['file_name']
    landmarks_df.drop('file_name', axis=1, inplace=True)
    landmarks_df.reset_index()
    gc.collect()

    for idx in tqdm(landmarks_df.index[start:end]):
        img_path = os.path.join(IMAGE_PATH, image_names_lst[idx])
        save_path = os.path.join(SAVE_PATH, image_names_lst[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points_lst = landmarks_to_points(landmarks_df.loc[idx, :])
        image = draw_landmarks(image, points_lst)

        plt.imshow(image)
        plt.savefig(save_path)
        
        if idx % 100 == 0:
            gc.collect()


def parallel_run() -> None:
    train_size = len(os.listdir(IMAGE_PATH))
    chunks_lst = list(zip(
        range(0, train_size, CHUNK_SIZE),
        range(CHUNK_SIZE, train_size, CHUNK_SIZE)
    ))
    print(chunks_lst)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(generate_imgs_with_landmarks, chunks_lst)

    print(f'All {len(chunks_lst)} processes finished')


if __name__ == '__main__':
    args = parse_arguments()

    if args.parallel:
        sys.exit(parallel_run())
    else:
        sys.exit(generate_imgs_with_landmarks())
