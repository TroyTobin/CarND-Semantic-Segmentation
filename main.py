import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import argparse
import sys
from enum import Enum
import scipy.misc
import numpy as np
from moviepy.editor import VideoFileClip

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CREATE_PLOTS              = True
MOMENTUM                  = 0.9
num_classes = 2
image_shape = (160, 576)
data_dir    = './data'
runs_dir    = './runs'
model_dir   = './model'

# run mode enum
class Mode(Enum):
    TRAIN    = 1
    CLASSIFY = 2
    UNKNOWN  = 255

# input file type enum
class FileType(Enum):
    VIDEO = 1
    IMAGE = 2
    UNKNOWN = 255


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load vgg model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_model = tf.get_default_graph()
    
    # load the layers we will be using
    input_layer = vgg_model.get_tensor_by_name(vgg_input_tensor_name)
    keep_propability_layer = vgg_model.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_layer = vgg_model.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_layer = vgg_model.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_layer = vgg_model.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return input_layer, keep_propability_layer, layer3_layer, layer4_layer, layer7_layer


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    layer3_conv1_1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4_conv1_1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer7_conv1_1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # strides (2, 2)
    model_connections  = tf.layers.conv2d_transpose(layer7_conv1_1, num_classes, 4, 2, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Create the new model by joining layers
    model_connections = tf.add(model_connections, layer4_conv1_1)

    # strides (2, 2)
    model_connections = tf.layers.conv2d_transpose(model_connections, num_classes, 4, 2, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    model_connections = tf.add(model_connections, layer3_conv1_1)

    # strides (8, 8)
    model_connections = tf.layers.conv2d_transpose(model_connections, num_classes, 16, 8, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return model_connections
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # the ouput tensor is 4D so we needs to 2D (image dimensions)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
   
    # classify and determine cross entropy from the logits
    ce_logits = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    loss = tf.reduce_mean(ce_logits)

    if ("Adam" in optimize.Optimizer):
        # create Adam optimiser
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif ("Adadelta" in optimize.Optimizer):
        # create Adadelta optimiser
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif ("Adagrad" in optimize.Optimizer):
        # create Adadelta optimiser
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif ("Momentum" in optimize.Optimizer):
        # create Adadelta optimiser
        optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
    elif ("GradientDescent" in optimize.Optimizer):
        # create Adadelta optimiser
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        print ("Unknown optimizer %s" % (optimize.Optimizer))
        sys.exit(1)

    # create the minimize operation - minimising for loss (error)
    min_op = optimizer.minimize(loss)

    return logits, min_op, loss
optimize.Optimizer = "GradientDescent"


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # store each epoch loss so it can be plotted for comparison
    losses = []
    final_loss = 1e6
    for epoch in range(epochs):
        images_labels = get_batches_fn(batch_size)

        epoch_loss = 0
        for image_label in images_labels:
            res = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image_label[0], 
                                                                      correct_label: image_label[1], 
                                                                      keep_prob: train_nn.kp, 
                                                                      learning_rate: train_nn.lr})
            epoch_loss += res[1]

        final_loss = epoch_loss
        print ("Epoch %d loss: %f" % (epoch, epoch_loss))
        losses.append(epoch_loss)

    # Plot the loss per epoch
    if (CREATE_PLOTS):
        fig, ax = plt.subplots()
        plt.bar(range(epochs), losses)
        plt.title("Loss for each epoch")
        plt.xticks(range(epochs), range(epochs))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print(train_nn.kp, train_nn.lr)
        plt.savefig('loss_vs_epoch_%f_%f.png' % (train_nn.kp, train_nn.lr))
    return final_loss
train_nn.kp = 0.5
train_nn.lr = 0.5


def processImage(image):
    """
    Process a single image through the trained model and perform the semantic segmentation

    NOTE: the bulk of this function reuses code from 'gen_test_output' in helpers.py, so internal function created instead
    """
    output_data = helper.process_image(processImage.sess, image, processImage.image_pl, processImage.logits, processImage.keep_prob, processImage.image_shape)
    return output_data
processImage.image_shape = None
processImage.sess = None
processImage.logits = None
processImage.keep_prob = None
processImage.image_pl = None

def processing(sess, model_file, mode, epochs, batch_size, kr, optimizer, lr, input_file, output_file, file_type):
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    
    # path to saved model
    if (model_file is None):
        model_path = os.path.join(data_dir, 'vgg_seg_model_%d_%d_%f_%f_%s'%(epochs, batch_size, kr, lr, optimizer))
    else:
        model_path = os.path.join(data_dir, model_file)

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    
    # templates for learning rates and labels
    learning_rate = tf.placeholder(tf.float32, [])
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

    # TODO: Build NN using load_vgg, layers, and optimize function
    optimize.Optimizer = optimizer

    input_, keep, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
    model = layers(layer3, layer4, layer7, num_classes)
    logits, train_op, cross_entropy_loss = optimize(model, correct_label, learning_rate, num_classes)

    # only train once
    final_loss = 0
    config = []
    if (mode == Mode.TRAIN):
        sess.run(tf.global_variables_initializer())
        train_nn.kp = kr
        train_nn.lr = lr
        final_loss = train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_, correct_label, keep, learning_rate)
        config = [epochs, batch_size, kr, lr, optimizer]
        saver = tf.train.Saver()
        saver.save(sess, model_path)

        # save the test images showing result of training the model
        helper.save_inference_samples(runs_dir, data_dir + '_test', sess, image_shape, logits, keep, input_)

    elif (mode == Mode.CLASSIFY):
        # load the semantic segmentation model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        # store useful state for the image processer
        processImage.image_shape = image_shape
        processImage.sess = sess
        processImage.logits = logits
        processImage.keep_prob = keep
        processImage.image_pl = input_

        if (file_type == FileType.VIDEO):
            # load video for processing
            #try:
                testVideo = VideoFileClip(input_file)
                output = testVideo.fl_image(processImage)
                output.write_videofile(output_file, audio=False)
            #except:
            #    print("ERROR: Procesing input video file")
            #    sys.exit(1)
        elif (file_type == FileType.IMAGE):
            # load image for processing
            try:
                image  = mpimg.imread(input_file)
                output = processImage(image)
                mpimg.imsave(output_file, output)
            except:
                print("ERROR: Processing input image file")
                sys.exit(1)
        else:
            print("ERROR: Unkown input file type")
            sys.exit(1)
    return (final_loss, config)

def run(mode, model_file, epochs, batch_size, kr, optimizer, lr, input_file, output_file, file_type):
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        best_loss = 1e6
        best_config = []
        if ('FindBest' in optimizer):
            for optimizer in ['Adam', 'Adagrad', 'Adadelta', 'GradientDescent', 'Momentum']:
                [loss, config] = processing(sess, model_file, mode, epochs, batch_size, kr, optimizer, lr, input_file, output_file, file_type)
                print ("Test:", loss, config)
                if (loss < best_loss):
                    best_loss = loss
                    best_config = config                
            print(best_config, best_loss)
        else:            
            processing(sess, model_file, mode, epochs, batch_size, kr, optimizer, lr, input_file, output_file, file_type)

            
def print_usage(errorMsg):
    print("ERROR: %s" %(errorMsg))
    print("usage: main.py --mode <train|classify> --model_file <file> --video <video file> --image <image file> --out_file <output file> --epochs <num> --batch_size <size> --keep_prob <prob> --optimizer <name> --learn_rate <rate>")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic Segmentation")
    parser.add_argument('--mode', help="Specify either 'train' or 'classify'")
    parser.add_argument('--model_file', help="Specify the model file when classifying")
    parser.add_argument('--epochs', help="Specify the number of epochs when training")
    parser.add_argument('--batch_size', help="Specify the batch size when training")
    parser.add_argument('--keep_prob', help="Specify the keep probability when training")
    parser.add_argument('--optimizer', help="Specify the optimizer when training.  Either 'Adam', 'Adadelta', 'Adagrad', 'Momentum' or 'GradientDecent'")
    parser.add_argument('--learn_rate', help="Specify the learning rate when training")
    parser.add_argument('--video', help="Specify a video file to run the semantic segmentation model on")
    parser.add_argument('--image', help="Specify an image file to run the semantic segmentation model on")
    parser.add_argument('--out_file', help="Specify a file to save the semantic segmentation output")
    args = parser.parse_args()

    mode = Mode.UNKNOWN
    if (args.mode is None):
        print_usage()
    elif (args.mode == 'train'):
        mode = Mode.TRAIN
    elif (args.mode == 'classify'):
        mode = Mode.CLASSIFY
    else:
        print_usage("No 'mode' provided")

    if (args.optimizer is not None):
        optimizer = args.optimizer
    else:
        print_usage("Need to specify optimizer")

    input_file    = None
    output_file   = None
    epochs        = None
    batch_size    = None
    keep_prob     = None
    learn_rate    = None
    file_type     = FileType.UNKNOWN
    model_file    = None

    if (mode == Mode.TRAIN):
        # need to have epoch, batch size, keep prob and learning rate specified
        if (args.epochs is not None):
            epochs = int(args.epochs)
        else:
            print_usage("Need to specify number of epochs")

        if (args.batch_size is not None):
            batch_size = int(args.batch_size)
        else:
            print_usage("Need to specify batch size")

        if (args.keep_prob is not None):
            keep_prob = float(args.keep_prob)
        else:
            print_usage("Need to specify number of keep probability")

        if (args.learn_rate is not None):
            learn_rate = float(args.learn_rate)
        else:
            print_usage("Need to specify learning rate")

    elif (mode == Mode.CLASSIFY):
        # Need to have an input and output specified
        if (args.video is not None):
            input_file = args.video
            file_type  = FileType.VIDEO
        elif (args.image is not None):
            input_file = args.image
            file_type  = FileType.IMAGE
        else:
            print_usage("No input data for semantic segmentation")

        # Need to have an output file to save the result
        if (args.out_file is not None):
            output_file = args.out_file
        else:
            print_usage("No output file to save the semantic segmentation result")

        # Need to have a model file specified
        if (args.model_file is not None):
            model_file = args.model_file
        else:
            print_usage("No model file to load")


    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
    tests.test_load_vgg(load_vgg, tf)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)
    run(mode, model_file, epochs, batch_size, keep_prob, optimizer, learn_rate, input_file, output_file, file_type)
