import os
import sys

import pandas as pd
import numpy as np
import albumentations as alb
import cv2


LANDMARK_LABELS_LST = ['Point_M' + str(i) for i in range(30)]
CROP_SIZE = 128
DATA_PATH = 'data'


def main():
    '''
    Read data, apply transforms and save results
    '''
    transform = alb.Compose(
        [
            alb.RandomCrop(width=CROP_SIZE, height=CROP_SIZE),
            alb.RandomBrightness(p=0.2),
        ],
        keypoint_params=alb.KeypointParams(
            format='xy',
            label_fields=LANDMARK_LABELS_LST,
            remove_invisible=False
        ),
    )
    landmarks_df = pd.read_csv(os.path.join(DATA_PATH, 'train'))



if __name__ == '__main__':
    sys.exit(main())
