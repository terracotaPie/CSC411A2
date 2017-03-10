from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import toimage
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle as cPickle

import os
from scipy.io import loadmat
gray()

t = int(time.time())
#t = 1454219613
print("t=", t)
random.seed(t)


M = loadmat("mnist_all.mat")

import tensorflow as tf

# added this
num_labels = 6
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

def print_dict(dict):
    for key in dict:
        print(key, len(dict[key]))


SAVED_IMGS = os.getcwd() + '/img_data/tf/actor_imgs.p'

def get_img_sets(n_train, n_val, n_test):
    f = open(SAVED_IMGS, 'rb')
    imgs = cPickle.load(f)
    f.close()

    training_set = {}
    validation_set = {}
    test_set = {}

    # initialize dictionaries
    for a in act:
        training_set[a] = []
        validation_set[a] = []
        test_set[a] = []

    used_imgs = []

    ## build the training, validation and test sets. Ensure that all sets are
    ## unique.
    # build the training set with N imgs per actor
    # N = n_train
    for a in act:
        for i in range(n_train):
            training_set[a].append(imgs[a][i][1])   # get the image
            used_imgs.append(imgs[a][i][0])     # get the filename of the image

    # build the validation set of 30 images
    i = 0
    for a in act:
        while i != n_val:
            if (imgs[a][i][0] not in used_imgs) == False:
                validation_set[a].append(imgs[a][i][1])   # get the image
                used_imgs.append(imgs[a][i][0])   # keep track of imgs used in another set already
                i += 1

    # build the test set with 30 validation sets per user
    i = 0
    for a in act:
        while i != n_test:
            if (imgs[a][i][0] not in used_imgs) == False:
                test_set[a].append(imgs[a][i][1])   # get the image
                used_imgs.append(imgs[a][i][0])     # get the filename of the image
                i += 1

    return training_set, validation_set, test_set

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
'''
    Return the x, y-vectors of the training set.
    Preprocess the images here by flattening and normalizing them.
'''
def preprocess(img_set):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_labels))

    imgs = {}

    for a in act:

        # shuffle the images for randomization
        imgs = img_set[a]
        random.shuffle(imgs)

        for img in imgs:
            # resize image to 28 x 28
            img2 = imresize(toimage(img), (28,28))
            x = np.array([])
            
            if img2.ndim == 2: # 2D image found
                # normalize the image
                x = img2 / 255.
            else:
                # greyscale and normalize the image
                x = rgb2gray(img2)

            y = zeros(6)
            y[act.index(a)] = 1
            # append the img vector and its label to the X and Y vector
            print("THIS", type(batch_xs))
            # np.vstack((batch_xs, x))
            # np.vstack((batch_y_s, y))

    return batch_xs, batch_y_s


'''
    for getting minibatch of size N from the training set
    https://piazza.com/class/iu4xr8zpnvo7k0?cid=490
'''
def get_train_batch(M, N):

    n = int(N/num_labels)
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_labels))

    train_k =  ["train"+str(i) for i in range(num_labels)]

    train_size = len(M[train_k[0]])
    #train_size = 5000

    for k in range(num_labels):
        train_size = len(M[train_k[k]])

        # generate random set of indices
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(num_labels)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s



## PART 7 CODE
'''
Notes:
- Part 9 weight visualization:
    6 filters, 1 filter to identify an actor
    weights of each filter = an actor's face



- preproccess images:
    convert images to grayscale
    normalize
    flatten into vector

    input layer, X =  num_imgs x 784
    hidden layer, # hidden layers: nhid = 300 units
                tanh activations

    output layer: 6 units, one per actor
        uses one-hot encoding
        output = softmax(output layer)

    - weight and bias initialization
'''


act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
PICKLED_act = os.getcwd() + "/img_data/actor_imgs.p"  # lol ew

# https://piazza.com/class/iu4xr8zpnvo7k0?cid=364
# initialize weights and biases to some random number
tf.set_random_seed(52515215)
x = tf.placeholder(tf.float32, [None, 784])

nhid = 300
img_size = 28*28
W0 = tf.Variable(tf.random_normal([784, nhid], stddev=0.01))  # first layer
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, num_labels], stddev=0.01))  # hidden layer
b1 = tf.Variable(tf.random_normal([num_labels], stddev=0.01))

# layer outputs
layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)   # hidden layer with tanh activations
layer2 = tf.matmul(layer1, W1)+b1          # output layer

# get target labels with softmax function
y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, num_labels])


## train the network with Adam
lam = 0.00000
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(reg_NLL)

## begin tensorflow session
#init = tf.initialize_all_variables()     # this line will be deprecated soon.
# replced tf.initialize_all_variables() with tf.gloable_variables_initializer()
# in line below
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

training_set, validation_set, test_set = get_img_sets(90, 30, 30)
test_x, test_y = preprocess(test_set)
validation_x, validation_y = preprocess(validation_set)

# store learning curve plot values
x_vals = []
training_acc = []
validation_acc = []
test_acc = []

# notes on placeholder values:
# http://learningtensorflow.com/lesson4/

## train the NN
#for i in range(5000):
for i in range(500):
    batch_xs, batch_ys = preprocess(training_set)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # output info at each step
    if i % 50 == 0:
        print()
        print("i=", i)
        # get values of tensor
        training_val = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        test_val = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        validation_val = sess.run(accuracy, feed_dict={x:validation_x, y_: validation_y})
        

        # record NN performance across training, test and validation sets
        # to graph later with the following variables n each axis:
        # x-axis: x_vals
        # y_axis: train_acc, test_acc, val_acc

        x_vals.append(i)
        training_acc.append(training_val)
        validation_acc.append(validation_val)
        test_acc.append(test_val)

        print("Train:", training_val)
        print("Validation:", validation_val)
        print("Test:", test_val)

# end TensorFlow session
sess.close()


## plot learning results
# plt.title('TensorFlow NN Learning curves on Training, Validation and Test Sets')
# plt.xlabel('Number of Iterations in Gradient Descent')
# plt.ylabel('Percentage of Correctly Predicted Labels')
# 
# # plot the learning curves
# plt.plot(x_vals, training_acc, 'ro-')
# plt.plot(x_vals, validation_acc, 'go-')
# plt.plot(x_vals, test_acc, 'bo-')
# 
# # save the graph in a png
# plt.savefig('p7_perf.png', bbox_inches='tight')


# 
# def part9():
#     # visualize weights for first actor
#     actor_1 = act[0]
#     dest_file = os.getcwd() + '/images/actor_1_weights.png'
#     y1 = [1, 0, 0, 0, 0, 0]
#     W1 = get_w1(y1)
#     W1_img = toimage(np.reshape(W1, (32, 32), 'bilinear'))
#     W1_img.save('')
# 
#     # visualize weights for second actor
#     actor_2 = act[3]
#     dest_file = os.getcwd() + '/images/actor_2_weights.png'
#     y2 = [0, 0, 0, 1, 0, 0]
#     W1 = get_w1(y1)
#     W2_img = toimage(np.reshape(W1, (32, 32), 'bilinear'))
