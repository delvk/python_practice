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


def log_print(something, log_file=None):
    """
    This print and log the same time
    """
    print(something)
    if log_file:
        log_file.write(something)


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
        mse, mae = mcnn.loss(d_map_estimation, d_map_placeholder)

        # define optimization algorithm
        optimizer = tf.train.AdamOptimizer(args.learning_rate)

        # training node
        train_op = optimizer.minimize(mse)

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
            train_log = open(join(session_path, "training_process.log"), "w+")
            train_loss_log = open(join(session_path, "train_loss.log"), "a+")
            val_loss_log = open(join(session_path, "val_loss.log"), "a+")
            # running session
            sess.run(init)
            if args.retrain:
                utils.load_weights(Graph, args.base_model_path)

            # start the epochs
            for epoch in range(args.num_epochs):
                # record starting time
                start_time = time.time()
                
                temp_str="=================================Training============================="
                log_print(temp_str)

                temp_str="Epoch: {}".format(epoch + 1)
                log_print(temp_str)

                # get the list of the train images and ground_truth
                train_img_list, train_d_map_list = utils.get_data_list(args.train_path)

                # define train_total_loss
                train_total_loss = 0

                # loop through all the training images
                number_of_images = len(train_img_list)
                for img_idx in range(number_of_images):
                    # load image
                    train_image = utils.load_image(train_img_list[img_idx])
                    # and ground truth
                    train_d_map = utils.load_ground_truth(
                        train_d_map_list[img_idx], downsize=True
                    )

                    # reshape the tensor before feeding it to the network
                    train_image = utils.reshape_tensor(train_image)
                    train_d_map = utils.reshape_tensor(train_d_map)

                    # prepare the feed_dictionary
                    feed_dict = {
                        image_placeholder: train_image,
                        d_map_placeholder: train_d_map,
                    }
                    # compute the loss of one image
                    _, loss_per_image = sess.run([train_op, mse], feed_dict=feed_dict)

                    # accumulate the loss all over the training images
                    train_total_loss += loss_per_image

                # ending of the for loop - mean end 1 epoch
                end_time = time.time()
                train_duration = end_time - start_time

                # compute the average training loss
                train_avg_loss = train_total_loss / number_of_images

                # Then we print the results for this epoch, also log it
                temp_str = "Epoch {}/{} took {:.3f}s - MSE: {:.4f}".format(
                    epoch + 1, args.num_epochs, train_duration, train_avg_loss
                )
                log_print(temp_str, log_file=train_log)

                # write to csv file
                train_loss_log.write("{}\n".format(train_avg_loss))
                # =================================================================================================================
                # Validate the model
                temp_str="--------------------------------Validation----------------------------------"
                log_print(temp_str)

                val_img_list, val_gt_list = utils.get_data_list(args.val_path)
                # define val_total_loss
                val_total_loss = 0

                # loop all validation images
                for i in range(len(val_img_list)):
                    # load image
                    val_image = utils.load_image(train_img_list[i])

                    # and ground truth
                    val_d_map = utils.load_ground_truth(val_gt_list[i], downsize=True)

                    # reshape the tensor for feeding it to the network
                    val_image = utils.reshape_tensor(val_image)
                    val_d_map = utils.reshape_tensor(val_d_map)

                    # Prepare the feed_dict
                    feed_dict_val = {
                        image_placeholder: val_image,
                        d_map_placeholder: val_d_map,
                    }
                    # compute the loss per image
                    loss_per_image = sess.run(mse, feed_dict=feed_dict_val)

                    # accumulate the validation loss across all the images.
                    val_total_loss += loss_per_image

                val_avg_loss = val_total_loss / len(val_img_list)

                # print and log the result
                temp_str = "Validation over {}, MSE = {}".format(
                    len(val_img_list), val_avg_loss
                )
                log_print(temp_str, train_log)

                # save the weights + bias and the summary
                utils.save_weights(
                    Graph, join(session_path, "weights.%s" % (epoch + 1))
                )
                summary_str = sess.run(summary, feed_dict=feed_dict_val)
                writer.add_summary(summary_str, epoch)

                # --------------------Testing----------------------------------------------------------
                image = utils.load_image('test/IMG_1.jpg')
                image = utils.reshape_tensor(image)
                start_time = time.time()
                feed_dict = {image_placeholder: image}
                predict = sess.run(d_map_estimation, feed_dict)

                # resize from (1,x,x,1) to (x,x)
                predict = np.reshape(
                    predict, newshape=(predict.shape[1], predict.shape[2])
                )
                gt_count = utils.load_ground_truth("test/IMG_1.csv", downsize=False)
                gt_count = np.sum(gt_count)
                et_count = np.sum(predict)

                end_time = time.time()
                print("Time to predict: {:.2f}".format(end_time - start_time))
                print("GT Count: {}".format(gt_count))
                print("Estimate Count: {}".format(et_count))
            train_log.close()
            # valid_log.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./log", type=str)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--learning_rate", default=0.00001, type=float)
    parser.add_argument(
        "--train_path",
        default="/home/jake/Desktop/Projects/Python/dataset/SH_B/cooked/train_10",
        type=str,
    )
    parser.add_argument(
        "--val_path",
        default="/home/jake/Desktop/Projects/Python/dataset/SH_B/cooked/val_10",
        type=str,
    )
    parser.add_argument("--session_id", default=3, type=int)
    parser.add_argument("--retrain", default=False, type=bool)
    parser.add_argument(
        "--base_model_path", default="log/session:2/weights.1.npz", type=str
    )
    args = parser.parse_args()
    main(args)
    # print('Hello world')
