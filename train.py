"""
This script suppose to do this things
1. Load data
2. Load mcnn model that predifine in src.mcnn
3. Training, save model with best result each epoch
"""
# import libary
import tensorflow as tf
import src.loader as loader
import src.layer as layer
import utils as utils
from skimage.color import rgb2gray
from skimage.data import imread
import numpy as np
import pandas as pd


def load_data(data_root, mode):
    image_list = []
    gt_list = []
    if mode == 'train':
        image_list, gt_list = utils.get_data_list(data_root, mode='train')
    elif mode == 'valid':
        image_list, gt_list = utils.get_data_list(data_root, mode='valid')
    else:
        image_list, gt_list = utils.get_data_list(data_root, mode='test')

    # convert images and index to array
    for image_index in range(len(image_list)):
        # load image and ground truth
        images = rgb2gray(np.asarray(imread(image_list[image_index]), dtype=np.float32))
        density_maps = np.asarray(pd.read_csv(
            gt_list[image_index], header=None), dtype=np.float32)
    return images, density_maps


def main(args):
    images, gt = load_data(args.data_root, mode='valid')
    print(images.shape)
    print(gt.shape)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data', type=str)
    args = parser.parse_args()
    main(args)
