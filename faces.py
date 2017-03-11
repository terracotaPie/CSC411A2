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


# from caffe_classes import class_names

# display images in grayscale
gray()

RESULTS_FOLDER = os.getcwd() + '/images'
SAVED_IMGS = os.getcwd() + '/img_data/tf/actor_imgs.p'
ACTORS = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']



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
    Return the set of images in a single input vector(X)
    and their one-hot encodings(Y).
'''
def get_xy_vectors(actor_to_imgs, i, n, k):
    batch_xs = zeros((0, n))
    batch_y_s = zeros((0, 6))
    
    for actor in ACTORS:
        # add the i'th actor
        img_set = actor_to_imgs[actor][i]
        batch_xs = vstack((batch_xs, img_set))
        y = zeros(k)
        y[ACTORS.index(actor)] = 1
        batch_y_s = vstack((batch_y_s, tile(y, (len(img_set), 1))))
    return batch_xs, batch_y_s



'''
    Preprocess the images before passing them through the neural network.
    Grayscale, flatten and normalize the images passed through.
'''
def preprocess(raw_actor_to_imgs):
    # BAD_IMGS is a list of images to remove from the image set. These images
    # do not contain a distinguishable face.
    BAD_IMGS = ['baldwin23.jpg', 'baldwin77.jpg', 'carell49.jpg','carell105.jpg',\
    'carell115.jpg','chenoweth14.jpg','chenoweth32.jpg','chenoweth92.jpg',\
    'chenoweth99.jpg','chenoweth109.jpg','drescher71.jpg','drescher98.jpg',\
    'drescher121.jpg','ferrera129.jpg','hader76.jpg','hader115.jpg']
    
    actor_to_imgs = {}
    
    for actor in raw_actor_to_imgs:
        img_list = []
        for tup in raw_actor_to_imgs[actor]:
            filename, img = tup

           ##   if (filename == BAD_IMGS[0]):
            #     print(filename not in BAD_IMGS)
            # filter out the bad imgs to not keep in the img_list
            if filename not in BAD_IMGS:
                # img = tup
                # print("IMAGE SHAPE-----------------------")
                # print("ORIGINAL", shape(img))
                # 
                
                # 2D image found
                if img.ndim == 2:
                    img = img / 255.
                else:
                    # grayscale the RGB image
                    img = rgb2gray(img)
                    # print("after grayscale", shape(img))
                
                # flatten the image into a vector
                img = np.ndarray.flatten(img)
                img_list.append(img.T)
            
        actor_to_imgs[actor] = img_list
    for actor in actor_to_imgs:
        img_list = actor_to_imgs[actor]
        
        # randomize the images picked for each of the 3 image sets(training,
        # testing, validation)
        random.shuffle(img_list)
        
        # The smallest number of images an actor has is around 150
        # There are 150 images to create the 3 image sets(training, 
        # validation, testing) from.
        # training set size = 90 imgs/actor
        # validation set size = 30 imgs/actor
        # test set size = 30 imgs/actor
        training_set = vstack(img_list[0:90])
        validation_set = vstack(img_list[90:120])
        test_set = vstack(img_list[120:150])
        
        actor_to_imgs[actor] = [training_set, validation_set, test_set]

   ##   for key in actor_to_imgs:
    #     print(key, len(actor_to_imgs[key][0]), len(actor_to_imgs[key][1]), len(actor_to_imgs[key][2]))

    return actor_to_imgs

'''
    Train a single-hidden layer neural network to classify images and detect
    which actor of the 6 in ACTORS it is.
    
    Report the accuracy of the neural network over 5000 iterations of training
    and plot the performance.
'''
def part7(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    tf.set_random_seed(48548514)
    x = tf.placeholder(tf.float32, [None, 3600])

    nhid = 300                # number of hidden units
    W0 = tf.Variable(tf.random_normal([3600, nhid], stddev=0.01))         # layer 1 weights
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))           # hidden layer weights
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

    # layer activations
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1

    # get target labels with softmax function
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])

    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    ## NOTE: ADJUST LEARNING STEP THRESHOLD HERE
    train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x_vals = []               # x axis values
    train_acc = []            # y axis values to plot
    validation_acc = []
    test_acc = []
    
    for i in range(5001):
      
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
    
    # end TensorFlow session
    sess.close()
    
    ## plot the part 7 code performance results and save
    clf()
    plt.figure(1)

    # set up graph labels
    plt.title('Learning Curve of TensorFlow Classifier on Image Sets')
    plt.xlabel('Iterations of Gradient Descent')
    plt.ylabel('Classification Accuracy(%)')
    
    # plot data
    plt.plot(x_vals, train_acc, 'g')
    plt.plot(x_vals, validation_acc, 'b')
    plt.plot(x_vals, test_acc, 'r')
    
    plt.legend(['Training Set', 'Validation Set', 'Test Set'], loc='lower right')
    
    
    # save results
    savefig(RESULTS_FOLDER + '/part7_results.png')



def part9(train_images, train_labels, val_images, val_labels, test_images, test_labels, nhid):

    tf.set_random_seed(31854692)
    x = tf.placeholder(tf.float32, [None, 3600])

    # nhid = 300
    W0 = tf.Variable(tf.random_normal([3600, nhid], stddev=0.01))         # layer 1 weights
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

    W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))           # hidden layer weights
    b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
    
    # layer activations
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1

    # get target labels with softmax function
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, 6])

    lam = 0.00001
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(NLL)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    x_vals = []               # x axis values
    train_acc = []            # y axis values to plot
    validation_acc = []
    test_acc = []
        
    print("Processing")
    for i in range(5001):
        
        # train network
        batch_xs, batch_ys = get_train_batch(train_images, train_labels, 50, len(train_images))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        
        # print progress every 100 iterations
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
        

    W0_ = sess.run(W0)
    W1_ = sess.run(W1)
    sess.close()
    
    # return the weights of the trained NN
    return W0_, W1_


'''

'''
def visualize_weights(W0, W1, i, num_hidden):
    clf()
    fig = plt.figure(1)

    # set title
    title_str = "Weights with {:d} Hidden Units".format(num_hidden)
    plt.title(title_str)
    
    axes = fig.gca()
    img = axes.imshow(W0[:,i].reshape((60,60)), cmap = cm.coolwarm)    
    fig.colorbar(img, shrink=0.5, aspect=5)
    
    # save the image
    savefig(RESULTS_FOLDER + '/' + "part9_results_{:d}_nh{:d}".format(i, num_hidden) + ".png")



if __name__ == '__main__':
    ## RUN CODE
    n = 60
    num_pixels = 3600
    num_labels = 6          # num_labels
    
    random.seed(554744718)
    
    
    # actor_imgs_NOFILENAME.p was generated by get_all_imgs2.py
    # It contains a dictionary of cropped actor faces in RGB format, stored in numpy arrays.
    # resize and normalize when necessary.
    
    raw_actor_to_imgs = cPickle.load(open(SAVED_IMGS, 'rb'))
    actor_to_imgs = preprocess(raw_actor_to_imgs)
    
    train_imgs, train_labels = get_xy_vectors(actor_to_imgs, 0, num_pixels, num_labels)
    val_imgs, val_labels = get_xy_vectors(actor_to_imgs, 1, num_pixels, num_labels)
    test_imgs, test_labels = get_xy_vectors(actor_to_imgs, 2, num_pixels, num_labels)
    
    ## PART 7
    print("BEGIN PART 7 ======================================================")
    # part7(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels)
  
 #    ## PART 9
    # use 300 hidden units
    print("BEGIN PART 9 ======================================================")
    print("On 300 hidden units -------------")
    W0_300, W1_300 = part9(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, 300)

    visualize_weights(W0_300, W1_300, 50, 300)
    visualize_weights(W0_300, W1_300, 100, 300)
    visualize_weights(W0_300, W1_300, 200, 300)
    visualize_weights(W0_300, W1_300, 299, 300)
    
    # use 600 hidden units
    print("On 600 hidden units -------------")
    W0_600, W1_600 = part9(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, 600)

    visualize_weights(W0_300, W1_300, 200, 600)
    
    # use 900 hidden units
    print("On 900 hidden units -------------")
    W0_900, W1_900 = part9(train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels, 900)

    visualize_weights(W0_300, W1_300, 200, 900)
    
