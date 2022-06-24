import glob
import os
import cv2
import numpy as np
import torch

img_path = "../sample_data/features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png"
folder_path = "../sample_data/features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1"


def img2vector(img_path):
    # shape: width * height * channel
    # 没有减去平均数，只归一到[-1, 1]
    # TODO: 可以减去平均数

    return np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255)



def video2vector(folder_path):
    img_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
    return [img2vector(img_path) for img_path in img_paths]


# mean_file = np.array([123, 117, 102], dtype=np.float32)
# print(img2vector(img_path) - mean_file[None, None, :])
