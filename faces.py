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
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# Provided course code from Guerzhoy's class site:
# http://www.cs.toronto.edu/~guerzhoy/411/proj1/rgb2gray.py
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

    
def get_train_batch(M, N):
    
    n = int(N/num_labels)
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_labels))
    
    train_k =  ["train"+str(i) for i in range(num_faces)]

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(num_labels):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(num_labels)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    

def get_test(M):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_faces))
    
    test_k =  ["test"+str(i) for i in range(num_faces)]
    for k in range(num_faces):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(num_faces)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s


def get_validation(M):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_faces))
    
    validation_k =  ["validation"+str(i) for i in range(num_faces)]
    for k in range(num_faces):
        batch_xs = vstack((batch_xs, ((array(M[validation_k[k]])[:])/255.)  ))
        one_hot = zeros(num_faces)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[validation_k[k]]), 1))   ))
    return batch_xs, batch_y_s
        

def get_train(M):
    batch_xs = zeros((0, 28*28))
    batch_y_s = zeros( (0, num_faces))
    
    train_k =  ["train"+str(i) for i in range(num_faces)]
    for k in range(num_faces):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(num_faces)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
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

'''
    Preprocess the x vector of images.
    For each image, grayscale, flatten and normalize.
'''
def preprocess_imgs(x):
    # ## preprocess the images
    # n = img.shape(1)[0]
    # new_img = rgb2gray(img)                # grayscale
    # new_img = (new_img - 127)/255          # normalize
    num_imgs = np.shape(x)[0]
    num_pixels = np.shape(x)[1]
    
    x_new = np.shape(x)
    for i in range(num_imgs):
        grayscaled = rgb2gray(x[i])
        grayscaled = flatten(grayscaled)
        grayscaled = (grayscaled - 127)/255
        np.append(x_new, grayscaled)
        
return x_new

# num_imgs = 6 * 100            # 100 imgs/actor, 6 actors present

# https://piazza.com/class/iu4xr8zpnvo7k0?cid=364
# initialize weights and biases to some random number
tf.random.seed(239782384)
x = tf.placeholder(tf.float32, [None, 784])
# 
# snapshot = cPickle.load(open("snapshot50.pkl", 'rb'), encoding="latin1")
# W0 = tf.Variable(snapshot["W0"])
# b0 = tf.Variable(snapshot["b0"])
# W1 = tf.Variable(snapshot["W1"])
# b1 = tf.Variable(snapshot["b1"])

nhid = 300
W0 = tf.Variable(tf.random_normal([784, nhid], stddev=0.01))  # first layer
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, num_faces], stddev=0.01))  # hidden layer
b1 = tf.Variable(tf.random_normal([num_faces], stddev=0.01))

# layer outputs
layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)   # hidden layer with tanh activations
layer2 = tf.matmul(layer1, W1)+b1          # output layer

# get target labels with softmax function
y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, num_faces])


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
test_x, test_y = get_test(M)
test_x = preprocess_imgs(test_x)

validation_x, validation_y = get_validation(M)
validation_x = preprocess_imgs(validation_x)

# store learning curve plot values
x_vals = []
training_acc = []
validation_acc = []
test_acc = []


## preprocess the images:load in img, grayscale, normalize and flatten it

# notes on placeholder values:
# http://learningtensorflow.com/lesson4/

## train the NN
#for i in range(5000):
for i in range(100):
  #print(i)  
  batch_xs, batch_ys = get_train_batch(M, 500)
  preprocess_imgs = batch_xs
  
  
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
  # output info at each step
  if i % 1 == 0:
    # get values of tensor
    training_val = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    validation_val = sess.run(accuracy, feed_dict={validation_x, y_: test_y})
    test_val = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    
    # record NN performance across training, test and validation sets
    # to graph later
    x_vals.append(i)
    lc_training_y.append(train_acc)
    lc_test_y.append(test_acc)
    
    print("i=",i)
    print("Test:", test_acc)
    batch_xs, batch_ys = get_train(M)

    print("Train:", train_acc)
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
plt.plot(x_vals, testing_acc, 'bo-')

# save the graph in a png
plt.savefig('p7_perf.png', bbox_inches='tight')