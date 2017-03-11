from pylab import *
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters
from scipy.io import loadmat
import urllib
import pickle as cPickle
import os
import time
import tensorflow as tf
from rgb2gray import *

from caffe_classes import class_names

# display images in grayscale
gray()

RESULTS_FOLDER = os.getcwd() + '/images'
# SAVED_IMGS = os.getcwd() + '/img_data/alexnet/actor_imgs_ALEXNET.p'
SAVED_IMGS = os.getcwd() + '/img_data/alexnet/actor_imgs.p'
ACTORS = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']



def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



'''
    Return the set of images in a single input vector(X)
    and their one-hot encodings(Y).
'''
def get_xy_vectors(actor_to_imgs, i, n, k):
    batch_xs = zeros((0, n))
    batch_y_s = zeros((0, k))

    for actor in ACTORS:
        # add the i'th actor
        img_set = actor_to_imgs[actor][i]

        batch_xs = vstack((batch_xs, img_set))
        y = zeros(k)
        y[ACTORS.index(actor)] = 1

        batch_y_s = vstack((batch_y_s, tile(y, (len(img_set), 1))))
    return batch_xs, batch_y_s



'''
    Return a batch of images of size N.
    This is for minibatching.
'''
def get_train_batch(imgset, labels, N, imgset_len):
    # randomize the images selected in this batch
    batch_indices = random.choice(imgset_len, N, replace=False)

    # get images and labels
    batch = imgset[batch_indices, :]
    batch_labels = labels[batch_indices, :]

    return batch, batch_labels


'''
    Process the images to input into a NN.
    get the 3D NumPy images, process them through AlexNet and get its
    conv4 activations of each actor, then normalize these and return them.
'''
def preprocess(raw_actor_to_imgs):
    # BAD_IMGS is a list of images to remove from the image set. These images
    # do not contain a distinguishable face.
    BAD_IMGS = ['baldwin23.jpg', 'baldwin77.jpg', 'carell49.jpg','carell105.jpg',\
    'carell115.jpg','chenoweth14.jpg','chenoweth32.jpg','chenoweth92.jpg',\
    'chenoweth99.jpg','chenoweth109.jpg','drescher71.jpg','drescher98.jpg',\
    'drescher121.jpg','ferrera129.jpg','hader76.jpg','hader115.jpg']
    
    actor_to_imgs = {}

    # gather 3D numpy images
    for actor in raw_actor_to_imgs:
        img_list = []
        for tup in raw_actor_to_imgs[actor]:
            filename, img = tup
          
            # filter out the bad imgs to not keep in the img_list
            if filename not in BAD_IMGS:
                # 3D image found keep this in the list of images
                if img.ndim == 3:
                    img_list.append(img.T)

        actor_to_imgs[actor] = img_list

    ## get the conv4 activations
    actor_to_imgs2 = get_alexnet_conv4(actor_to_imgs)
    for actor in actor_to_imgs2:
        img_list = actor_to_imgs2[actor]

        # randomize the images picked for each of the 3 image sets(training,
        # testing, validation)
        random.shuffle(img_list)

        # The smallest number of images an actor has is around 150
        # There are 150 images to create the 3 image sets(training,
        # validation, testing) from.
        # training set size = 90 imgs/actor
        # validation set size = 30 imgs/actor
        # test set size = 30 imgs/actor

        training_set = vstack([i/255. for i in img_list[0:90]])
        validation_set = vstack([i/255. for i in img_list[90:120]])
        test_set = vstack([i/255. for i in img_list[120:150]])


        actor_to_imgs2[actor] = [training_set, validation_set, test_set]

    return actor_to_imgs2



'''
    Get the conv4 activations of AlexNet after training it on the 6 actors' images.
'''
def get_alexnet_conv4(actor_to_imgs):
    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]
    x_rand = (random.random((1,) + xdim)/255.).astype(float32)
    img = x_rand.copy()

    # load in weights
    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding='latin1').item()
    
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # get conv4 activations per actor
    actor_to_imgs2 = {}
    for actor in actor_to_imgs:

        print("Processing conv4 activations for: ", actor)

        img_list = []
        for img0 in actor_to_imgs[actor]:

            img[0,:,:,:] = (img0.T[:,:,:3]).astype(float32)
            img = img - mean(img)

            # save the conv4 activation
            conv4_output = sess.run(conv4, feed_dict={x:img})

            # flatten and cast to numpy array
            processed = array(conv4_output).flatten()
            img_list.append(processed)

        actor_to_imgs2[actor] = img_list

    sess.close()


    return actor_to_imgs2



'''
    Pass the conv4 activations to feed into a NN of 300 hidden units and train it
    to classify an image for any actor in ACTORS.

    ACTORS = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', \
             'Alec Baldwin', 'Bill Hader', 'Steve Carell']
'''
def part10(train_images, train_labels, val_images, val_labels, test_images, test_labels):

    tf.set_random_seed(52635486)

    x = tf.placeholder(tf.float32, [None, 64896])

    nhid = 300
    W0 = tf.Variable(tf.random_normal([64896, nhid], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1

    W = tf.Variable(tf.random_normal([64896, 6], stddev=0.01))
    b = tf.Variable(tf.random_normal([6], stddev=0.01))
    layer = tf.matmul(x, W)+b

    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])

    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x_vals = []
    test_acc = []
    validation_acc = []
    train_acc = []
    for i in range(501):
        batch_xs, batch_ys = get_train_batch(train_images, train_labels, 50, len(train_images))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # print progress every 50 iterations
        if i % 50 == 0:
            print('\nitr = {:d}'.format(i))

            train_results = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels})
            validation_results = sess.run(accuracy, feed_dict={x: val_images, y_: val_labels})
            test_results = sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})

            print("Accuracy on:")
            print('    Training set', train_results)
            print('    Validation set:', validation_results)
            print('    Test set:', test_results)

            # record NN performance
            x_vals.append(i)
            train_acc.append(train_results)
            validation_acc.append(validation_results)
            test_acc.append(test_results)

    W0_final = sess.run(W0)
    W1_final = sess.run(W1)
    b0_final = sess.run(b0)
    b1_final = sess.run(b1)
    sess.close()

    ## plot performance
    clf()
    plt.figure(1)
    plt.title('NN with AlexNet Conv4 Activations: Learning Curve across Image Sets')
    plt.xlabel('Number of iterations of training')
    plt.ylabel('Classification Accuracy(%)')

    plt.plot(x_vals, train_acc)
    plt.plot(x_vals, validation_acc)
    plt.plot(x_vals, test_acc)
    plt.legend(['Training Set', 'Validation Set', 'Test Set'], loc='lower right')

    savefig(RESULTS_FOLDER + '/part10_perf.png')

    return W0_final, W1_final, b0_final, b1_final



if __name__ == '__main__':

    ## run PART 10
    num_labels = 6

    random.seed(21342342)

    raw_actor_to_imgs = cPickle.load(open(SAVED_IMGS, "rb"), encoding='latin1')
    actor_to_imgs = preprocess(raw_actor_to_imgs)

    training_imgs, training_labels = get_xy_vectors(actor_to_imgs, 0, 64896, num_labels)
    validation_imgs, validation_labels = get_xy_vectors(actor_to_imgs, 1, 64896, num_labels)
    test_imgs, test_labels = get_xy_vectors(actor_to_imgs, 2, 64896, num_labels)

    W0, W1, b0, b1 = part10(training_imgs, training_labels, validation_imgs, validation_labels, test_imgs, test_labels)
