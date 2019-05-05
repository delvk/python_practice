# Import library
from os.path import join
import os
import glob
import random
def get_file_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def get_data_list(root_path, mode='valid'):
    """
    - Input: data root
    - Output: 2 list contain path to images, grouth truth using for training, validation and testing

    @param data_root: root to data
    @param mode: 'train', 'valid' or 'test'
    """

    if mode == 'train':
        image_path = join(root_path, 'train_validation_data',
                          'train', 'images')
        ground_truth_path = join(
            root_path, 'train_validation_data', 'train', 'ground_truth')
    elif mode == 'valid':
        image_path = join(root_path, 'train_validation_data',
                          'valid', 'images')
        ground_truth_path = join(root_path, 'train_validation_data',
                                 'valid', 'ground_truth')
    else:
        image_path = join(root_path, 'test_data', 'images')
        ground_truth_path = join(root_path, 'test_data', 'ground_truth')

    image_list = [file for file in glob.glob(join(image_path,'*.jpg'))]
    gt_list = []
    for filepath in image_list:
        file_id = get_file_id(filepath)
        gt_file_path = join(ground_truth_path, 'ground_truth'+ file_id + '.csv')
        gt_list.append(gt_file_path)
    
    xy = list(zip(image_list, gt_list))
    random.shuffle(xy)
    s_image_list, s_gt_list = zip(*xy)

    return s_image_list, s_gt_list

# def create_session(log_dir, session_id):
#     """
#     Module to create a session folder. It will create a folder with a proper session
#     id and return the session path.
#     :param log_dir: root log directory
#     :param session_id: ID of the session
#     :return: path of the session id folder
#     """
#     folder_path = os.path.join(log_dir, 'session:'+str(session_id))
#     if os.path.exists(folder_path):
#         print ('Session already taken. Please select a different session id.')
#         sys.exit()
#     else:
#         os.makedirs(folder_path)
#     return folder_path