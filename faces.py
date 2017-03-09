from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle as cPickle

import os
from scipy.io import loadmat


t = int(time.time())
#t = 1454219613
print("t=", t)
random.seed(t)


M = loadmat("mnist_all.mat")

import tensorflow as tf

# added this
num_labels = 6
ACTORS =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


SAVED_IMGS = os.getcwd() + '/img_data/tf/actor_imgs.p'
def get_img_sets(train_imgs_per_actor, val_imgs_per_actor, test_imgs_per_actor):
    f = open(SAVED_IMGS, 'rb')
    imgs = cPickle.load(f)
    f.close()
    
    training_set = {}
    validation_set = {}
    test_set = {}
    shuffled_imgs = {}
    
    # initialize dictionaries
    for a in ACTORS:
        training_set[a] = []
        validation_set[a] = []
        test_set[a] = []
        
        # shuffle the images to randomize them
        shuffled_imgs[a] = shuffle(imgs[a])
    
    used_imgs = []

    ## build the training, validation and test sets. Ensure that all sets are 
    ## unique.
    # about 150 images per actor
    # 30 training
    # 120 remaining
    # 30 validation
    # 90 training
    
    # build the training set of 90 images
    for a in ACTORS:
        for i in range(train_imgs_per_actor):
            training_set[a].append(imgs[a][i][1])   # get the image
            used_imgs.append(imgs[a][i][0])     # get the filename of the image
    print("len(used_imgs)", len(used_imgs))
    # build the validation set of 30 images
    for a in ACTORS:
        for i in range(val_imgs_per_actor):
            if (imgs[a][i][0] not in used_imgs) == False:
                validation_set[a].append(imgs[a][i][1])   # get the image
                used_imgs.append(imgs[a][i][0])   # keep track of imags used in another set already
    
    # build the test set with 30 validation sets per user
    for a in ACTORS:
        for i in range(test_imgs_per_actor):
            if (imgs[a][i][0] not in used_imgs) == False:
                test_set[a].append(imgs[a][i][1])   # get the image
                used_imgs.append(imgs[a][i][0])     # get the filename of the image
    
    return training_set, validation_set, test_set


# '''
#     Get the entire training set.
#    This is the old function from digits_tf
# '''
# def get_train(M):
#     batch_xs = zeros((0, 28*28))
#     batch_y_s = zeros( (0, num_labels))
#     
#     train_k =  ["train"+str(i) for i in range(num_labels)]
#     for k in range(num_labels):
#         batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
#         one_hot = zeros(num_labels)
#         one_hot[k] = 1
#         batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
#     return batch_xs, batch_y_s

#     
# 
# def get_test(M):
#     batch_xs = zeros((0, 28*28))
#     batch_y_s = zeros( (0, num_labels))
#     
#     test_k =  ["test"+str(i) for i in range(num_labels)]
#     for k in range(num_labels):
#         batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
#         one_hot = zeros(num_labels)
#         one_hot[k] = 1
#         batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
#     return batch_xs, batch_y_s

'''
    Return the x, y-vectors of the training set.
    Preprocess the images here by flattening and normalizing them.
'''
def preprocess(img_set):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_labels))
    print("IN PREPROCESS")
    
    for a in ACTORS:
        for img in img_set[a]:
            # normalize the image
            preprocessed_img = (ndarray.flatten(np.array(img))-127)/255
            one_hot = zeros(6)
            one_hot[ACTORS.index(a)] = 1
            # append the img vector and its label to the X and Y vector
            batch_xs = vstack((batch_xs, preprocessed_img))
            batch_y_s = vstack((batch_y_s, one_hot))

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


ACTORS = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
PICKLED_ACTORS = os.getcwd() + "/img_data/actor_imgs.p"  # lol ew


# num_imgs = 6 * 100            # 100 imgs/actor, 6 actors present

# https://piazza.com/class/iu4xr8zpnvo7k0?cid=364
# initialize weights and biases to some random number

x = tf.placeholder(tf.float32, [None, 784])
# 
# snapshot = cPickle.load(open("snapshot50.pkl", 'rb'), encoding="latin1")
# W0 = tf.Variable(snapshot["W0"])
# b0 = tf.Variable(snapshot["b0"])
# W1 = tf.Variable(snapshot["W1"])
# b1 = tf.Variable(snapshot["b1"])

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

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

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
batch_xs, batch_ys = preprocess(training_set)
test_x, test_y = preprocess(test_set)
validation_x, validation_y = preprocess(validation_set)

# store learning curve plot values
x_vals = []
training_acc = []
validation_acc = []
test_acc = []

sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## preprocess the images:load in img, grayscale, normalize and flatten it

# notes on placeholder values:
# http://learningtensorflow.com/lesson4/

## train the NN
#for i in range(5000):
for i in range(5000):
    print("i=",i)
    
    # output info at each step
    if i % 1 == 0:
        # get values of tensor
        test_val = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        validation_val = sess.run(accuracy, feed_dict={x:validation_x, y_: validation_y})
        training_val = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        
        # record NN performance across training, test and validation sets
        # to graph later with the following variables per axis:
        # x-axis: x_vals
        # y_axis: train_acc, test_acc, val_acc
        #
        x_vals.append(i)
        training_acc.append(training_val)
        validation_acc.append(validation_val)
        test_acc.append(test_val)
        
        

        print("Train:", training_val)
        print("Validation:", validation_val)
        print("Test:", test_val)
        
        print("Penalty:", sess.run(decay_penalty))

## end TensorFlow session
sess.close()


## plot learning results
# set titles
plt.title('Learning curves on Training, Validation and Test Sets')
plt.xlabel('Number of images trained on')
plt.ylabel('Percentage of Correctly Predicted Labels')

# plot the learning curves
plt.plot(x_vals, training_acc, 'ro-')
plt.plot(x_vals, validation_acc, 'go-')
plt.plot(x_vals, test_acc, 'bo-')

# save the graph in a png
plt.savefig('p7_perf.png', bbox_inches='tight')

def part9():
    # visualize weights for first actor
    actor_1 = act[0]
    dest_file = os.getcwd() + '/images/actor_1_weights.png'
    y1 = [1, 0, 0, 0, 0, 0]
    W1 = get_w1(y1)
    W1_img = toimage(np.reshape(W1, (32, 32), 'bilinear'))
    W1_img.save('')
    
    # visualize weights for second actor
    actor_2 = act[3]
    dest_file = os.getcwd() + '/images/actor_2_weights.png'
    y2 = [0, 0, 0, 1, 0, 0]
    W1 = get_w1(y1)
    W2_img = toimage(np.reshape(W1, (32, 32), 'bilinear'))