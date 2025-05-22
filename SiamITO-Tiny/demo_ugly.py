import glob
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from dataset import center_crop

seq_path = sorted(glob.glob('./real_video/*'))
cv2.namedWindow('src', 0)
cv2.namedWindow('dst', 0)
clahe = cv2.createCLAHE(3, (8, 8))

for index in range(len(seq_path)):
    img_paths = sorted(glob.glob(os.path.join(seq_path[index], '*')))
    init = False
    x_t, y_t = 0, 0

    for img_path in img_paths:
        img = cv2.imread(img_path, 0)

        # img enhancement
        # dst = cv2.equalizeHist(img)
        dst = clahe.apply(img)

        cv2.imshow('src', img)
        cv2.imshow('dst', dst)

        cv2.waitKey(0)

