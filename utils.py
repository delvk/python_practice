# Import library
from os.path import join
import os
import glob
import random
import sys
import numpy as np
import tensorflow as tf
from skimage.transform import rescale, resize
from skimage.data import imread
from skimage.color import rgb2gray


def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def get_data_list(
    root_path, img_name="images", gt_name="ground_truth", extension="jpg"
):
    """
    - Input: data root
    - Output: 2 list contain path to images, grouth truth using for training, validation and testing

    @param data_root: root to data
    """
    extension = "*." + extension
    image_path = join(root_path, img_name)
    ground_truth_path = join(root_path, gt_name)
    image_list = [file for file in glob.glob(join(image_path, extension))]
    gt_list = []
    
    for filepath in image_list:
        file_id = get_file_id(filepath)
        gt_file_path = join(ground_truth_path, file_id + ".csv")
        gt_list.append(gt_file_path)

    xy = list(zip(image_list, gt_list))
    random.shuffle(xy)
    s_image_list, s_gt_list = zip(*xy)

    return s_image_list, s_gt_list


def create_session(log_dir, session_id):
    """
    Module to create a session folder. It will create a folder with a proper session
    id and return the session path.
    :param log_dir: root log directory
    :param session_id: ID of the session
    :return: path of the session id folder
    """
    folder_path = os.path.join(log_dir, "session:" + str(session_id))
    if os.path.exists(folder_path):
        print("Session already taken. Please select a different session id.")
        # sys.exit()
    else:
        os.makedirs(folder_path)
    return folder_path


def reshape_tensor(tensor):
    """
    Reshapes the input tensor appropriate to the network input
    i.e. [1, tensor.shape[0], tensor.shape[1], 1]
    :param tensor: input tensor
    :return: reshaped tensor
    """
    r_tensor = np.reshape(tensor, newshape=(1, tensor.shape[0], tensor.shape[1], 1))
    return r_tensor


def down_size_density_map(d_map):
    width = d_map.shape[0]
    height = d_map.shape[1]
    width_1 = width / 4
    height_1 = height / 4
    d_map = resize(d_map, (width_1, height_1), anti_aliasing=True)
    d_map = d_map * ((width * height) / (width_1 * height_1))
    return d_map


def save_weights(graph, fpath):
    """
    Module to save the weights of the network into a numpy array.
    Saves the weights in .npz file format
    :param graph: Graph whose weights needs to be saved.
    :param fpath: filepath where the weights needs to be saved.
    :return:
    """
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    variable_names = [v.name for v in variables]
    kwargs = dict(zip(variable_names, sess.run(variables)))
    np.savez(fpath, **kwargs)


def load_weights(graph, fpath):
    """
    Load the weights to the network. Used during transfer learning and for making predictions.
    :param graph: Computation graph on which weights needs to be loaded
    :param fpath: Path where the model weights are stored.
    :return:
    """
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    data = np.load(fpath)
    for v in variables:
        if v.name not in data:
            print("could not load data for variable='%s'" % v.name)
            continue
        print("assigning %s" % v.name)
        sess.run(v.assign(data[v.name]))


def load_image(path):
    """
    Load rgb image into gray_norm array
    """
    rgb_image = imread(path)
    gray_image = rgb2gray(rgb_image)
    train_norm = gray_image / np.max(gray_image)
    train_image_arr = np.asarray(train_norm, dtype=np.float32)

    return train_image_arr


def load_ground_truth(path, downsize=True):
    """
    Load csv ground truth to array
    """
    raw_csv = pd.read_csv(path, sep=",", header=None)
    arr = np.asarray(raw_csv, dtype=np.float32)
    if downsize:
        wd = int(arr.shape[0] / 4)
        ht = int(arr.shape[1] / 4)
        resize(arr, (wd, ht), anti_aliasing=True)

    return arr
