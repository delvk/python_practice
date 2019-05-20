"""
This script suppose to do this things
1. Load data
2. Load mcnn model that predifine in src.mcnn
3. Training, save model with best result each epoch
"""
# import libary
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.data import imread
import numpy as np
import pandas as pd
from os.path import join
import time

# my library
import src.loader as loader

# import src.layer as layer
import src.mcnn as mcnn
import utils as utils


def load_data(data_root, mode):
    """
     The images are built from tensor which must be 4-D with shape [batch_size, height, width, channels]
    """
    image_list = []
    gt_list = []
    if mode == "train":
        image_list, gt_list = utils.get_data_list(data_root, mode="train")
    elif mode == "valid":
        image_list, gt_list = utils.get_data_list(data_root, mode="valid")
    else:
        image_list, gt_list = utils.get_data_list(data_root, mode="test")

    # convert images and index to array
    for image_index in range(len(image_list)):
        # load image and ground truth
        images = rgb2gray(np.asarray(imread(image_list[image_index]), dtype=np.float32))
        d_maps = np.asarray(
            pd.read_csv(gt_list[image_index], header=None), dtype=np.float32
        )
    return images, d_maps


def main(args):
    print("==================================================")
    session_path = utils.create_session(args.log_dir, args.session_id)
    tf.random.set_random_seed(1997)
    Graph = tf.Graph()

    with Graph.as_default():
        # create images and density map holder
        image_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 1])
        d_map_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 1])

        # build all the nodes of the network mcnn
        d_map_estimation = mcnn.build_network(image_placeholder)

        # define loss function
        euc_loss = mcnn.loss(d_map_estimation, d_map_placeholder)

        # define optimization algorithm
        optimizer = tf.train.AdamOptimizer(args.learning_rate)

        # training node
        train_op = optimizer.minimize(euc_loss)

        # INITIAL ALL VARIABLES
        init = tf.group(
            tf.global_variables_initializer(), tf.local_variables_initializer()
        )

        # summary
        summary = tf.summary.merge_all()

        # START TRAINING SESSION
        with tf.Session(graph=Graph) as sess:

            # initialize the logging in this session
            writer = tf.summary.FileWriter(join(session_path, "training.log"))
            writer.add_graph(sess.graph)
            train_log = open("training_process.log", "w+")

            # running session
            sess.run(init)
            if args.retrain:
                utils.load_weights(Graph, args.base_model_path)

            # start the epochs
            for epoch in range(args.num_epochs):
                # record starting time
                start_time = time.time()
                print("Epoch: {}".format(epoch + 1))
                
                # get the list of the train images and ground_truth
                images_list, d_maps_list = utils.get_data_list(args.root_path)

                # define total_train_loss
                total_train_loss = 0

                # loop through all the training images
                for image_index in range(len(images_list)):
                    # load image
                    train_image = utils.load_image(images_list[image_index])
                    # and ground truth
                    train_d_map = utils.load_ground_truth(
                        d_maps_list[image_index], downsize=True
                    )

                    # reshape the tensor before feeding it to the network
                    train_image = utils.reshape_tensor(train_image)
                    train_d_map = utils.reshape_tensor(train_d_map)

                    # prepare the feed_dictionary
                    feed_dictionary = {
                        image_placeholder: train_image,
                        d_map_placeholder: train_d_map,
                    }
                    # compute the loss of one image
                    _, loss_per_image = sess.run(
                        [train_op, euc_loss], feed_dict=feed_dictionary
                    )

                    # accumulate the loss all over the training images
                    total_train_loss += loss_per_image

                # ending of the for loop - mean end 1 epoch
                end_time = time.time()
                train_duration = end_time - start_time

                # compute the average training loss
                avg_train_loss = total_train_loss

                # Then we print the results for this epoch, also log it
                print(
                    "Epoch {} of {} took {:.3f}s".format(
                        epoch + 1, args.num_epochs, train_duration
                    )
                )
                train_log.write(
                    "\nEpoch {} of {} took {:.3f}s".format(
                        epoch + 1, args.num_epochs, train_duration
                    )
                )
                print("  Training loss:\t\t{:.6f}".format(avg_train_loss))
                train_log.write("\n  Training loss:\t\t{:.6f}".format(avg_train_loss))

                # validating the model, we do this every epoch, feel free to config it for your purpose
                val_images_list, val_gts_list = utils.get_data_list(
                    args.root_path, mode="valid"
                )
                val_start_time = time.time()

                # define val_total_loss
                total_val_loss = 0

                # loop all validation images
                for i in range(len(val_images_list)):
                    # load image
                    val_image = rgb2gray(
                        np.asarray(imread(images_list[i]), dtype=np.float32)
                    )

                    # and ground truth
                    val_d_map = np.asarray(
                        pd.read_csv(val_gts_list[i], header=None), dtype=np.float32
                    )
                    val_d_map = utils.down_size_d_map(val_d_map)

                    # reshape the tensor for feeding it to the network
                    val_image = utils.reshape_tensor(val_image)
                    val_d_map = utils.reshape_tensor(val_d_map)

                    # Prepare the feed_dict
                    feed_dict_data = {
                        image_placeholder: val_image,
                        d_map_placeholder: val_d_map,
                    }
                    # compute the loss per image
                    loss_per_image = sess.run(euc_loss, feed_dict=feed_dict_data)

                    # accumulate the validation loss across all the images.
                    total_val_loss += loss_per_image
                val_end_time = time.time()
                val_duration = val_end_time - val_start_time

                avg_val_loss = total_val_loss

                # print and log the result
                print("  Validation loss:\t\t{:.6f}".format(avg_val_loss))
                print(
                    "Validation over {} images took {:.3f}s".format(
                        len(val_images_list), val_duration
                    )
                )
                print("===========================================================")
                train_log.write(
                    "\n==========================================================="
                )

                # save the weights + bias and the summary
                utils.save_weights(
                    Graph, join(session_path, "weights.%s" % (epoch + 1))
                )
                summary_str = sess.run(summary, feed_dict=feed_dict_data)
                writer.add_summary(summary_str, epoch)

                # testing
                image = imread("7_1.jpg", as_gray=True)
                image = utils.reshape_tensor(image)
                start_time = time.time()
                feed_dict = {image_placeholder: image}
                predict = sess.run(d_map_estimation, feed_dict)

                # resize from (1,x,x,1) to (x,x)
                predict = np.reshape(
                    predict, newshape=(predict.shape[1], predict.shape[2])
                )
                count = np.sum(predict[:])
                end_time = time.time()
                print("Time to predict: {}".format(end_time - start_time))
                print("Count: {}".format(count))
            train_log.close()
            # valid_log.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./log", type=str)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument(
        "--root_path",
        default="/home/jake/Desktop/Projects/Python/dataset/SH_B/cooked/train_10",
        type=str,
    )
    parser.add_argument("--session_id", default=2, type=int)
    parser.add_argument("--retrain", default=False, type=bool)
    parser.add_argument(
        "--base_model_path", default="log/session:2/weights.1.npz", type=str
    )
    args = parser.parse_args()
    main(args)
