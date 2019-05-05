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
def load_data(data_root):
    val_image_list, val_gt_list = utils.get_data_list(data_root, mode='valid')
    images = 0
    density_maps = 0
    return images, density_maps



def main(args):
    load_data(args.data_root)
    print('hello world')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data', type=str)
    args = parser.parse_args()
    main(args)