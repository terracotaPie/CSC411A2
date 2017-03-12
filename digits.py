
# coding: utf-8

# In[3]:

#get_ipython().magic('matplotlib inline')
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

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
#imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
#show()


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y) + b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    dCdL1 = y - y_
    dCdW1 = dot(L0, dCdL1.T)
    dCdB  = dot(W1, dCdL1)
    diff  = 1 - L0**2

    dCdW0 = tile(dCdB, 28 * 28).T * dot(x, diff.T)
    dCdb0 = dCdB * diff

    return dCdW1, dCdL1, dCdW0, dCdb0


# ## Part 1 - Dataset Description
#   * variety of angles
#   * different styles of handwriting
#   * gaps between continuous lines
#   * different thickness levels

# In[32]:
print("generating part 1")
f, sub = plt.subplots(10, 10, figsize=(15, 15))
for i in range(10):
    for j in range(10):
        sub[i][j].axis('off')
        sub[i][j].imshow(
            M['train' + str(i)][j + 20].reshape((28, 28)), cmap=cm.gray)

plt.savefig("images/dataset.png")


# ## Part 2 Compute network function

# In[5]:

def compute_network(x, W0, b0, W1, b1):
    _,_, output = forward(x, W0, b0, W1, b1)
    return argmax(output)


# ## Part 3 Cost function

# In[6]:

def cross_entropy(y, y_):
    return -sum(y_ * log(y))


# In[7]:

#Load sample weights for the multilayer neural network
def load_sample_weights():
    f = open("snapshot50.pkl", "rb")
    snapshot = cPickle.load(open("snapshot50.pkl", "rb"), encoding="latin1")
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300, 1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10, 1))
    return W0,b0,W1,b1

# L0, L1, output = forward(x, W0, b0, W1, b1)
# y = argmax(output)

# y_true = array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
# print(deriv_multilayer(W0, b0, W1, b1,x, L0, L1, output, y_true))
################################################################################
# Code for displaying a feature from the weight matrix mW
# fig = figure(1)
# ax = fig.gca()
# heatmap = ax.imshow(W0[:, 50].reshape((28, 28)), cmap=cm.coolwarm)
# fig.colorbar(heatmap, shrink=0.5, aspect=5)
# show()
################################################################################


# In[8]:

def get_batch(offset,example_per_class=5):
    # 5 examples per class
    classes_num = 10
    x_batch = np.zeros((example_per_class * classes_num, 28 * 28))
    y_batch = np.zeros((example_per_class * classes_num, classes_num))
    for i in range(classes_num):
        for j in range(example_per_class):
            x_batch[i * example_per_class + j] = M['train' + str(i)][j +
                                                                     offset]
            y_batch[i * example_per_class + j][i] = 1
    return x_batch, y_batch, example_per_class * classes_num


# In[9]:

#Load one example from the training set, and run it through the
def test_image(x):
    _,_, output = forward(x, W0, b0, W1, b1)
    return argmax(output)

def test_perf():
    hit = miss = 0
    for i in range(10):
        for image in M["test" + str(i)]:
            result = test_image(image.reshape(784,1))
            if result == i:
                hit+=1
            else:
                miss+=1
    return (float(hit)/float(hit + miss) * 100)


# In[10]:

# Do gradient descent
def train(plot=False):
    global W0, b0, W1, b1
    global plot_iters, plot_performance
    plot_iters = []
    plot_performance = []
    alpha = 1e-3
    for i in range(150):
        X, Y, examples_n = get_batch(i * 5,10)

        update = np.zeros(4)

        for j in range(examples_n):
            y = Y[j].reshape((10, 1))
            x = X[j].reshape((28 * 28, 1)) / 255.
            L0, L1, output = forward(x, W0, b0, W1, b1)
            gradients = deriv_multilayer(W0, b0, W1, b1, x, L0, L1, output, y)
            update = [update[k] + gradients[k] for k in range(len(gradients))]
            if (i * examples_n + j) % 500 == 0: 
                print("Iter %d" % (i * examples_n + j))

        # update the weights 
        W1 -= alpha * update[0]
        b1 -= alpha * update[1]
        W0 -= alpha * update[2]
        b0 -= alpha * update[3]
        if plot:
            plot_iters.append(i * examples_n)
            plot_performance.append(test_perf())
    return plot_iters,plot_performance


# In[33]:

plot_iters = []
plot_performance = []
W0,b0,W1,b1 = load_sample_weights()
print("Part 4 - Training the network")
train(plot=False)
test_perf()


# In[35]:

f, sub = plt.subplots(10,10, figsize=(15, 15))
for i in range(10):
    for j in range(10):
        sub[i][j].axis('off')
        sub[i][j].imshow(W0.T[i * 10 + j].reshape((28,28)))

plt.savefig("images/weights.png")
print("visualized weight and saved images/weights.png")


# In[13]:
print("Part 5 - Plotting performance graph")
skip = True
if not skip:
    W0,b0,W1,b1 = load_sample_weights()
    plot_iters,plot_performance = train(plot=True)

    plt.figure()
    plt.plot(plot_iters,plot_performance)
    plt.ylabel('performance')
    plt.xlabel('iterations')
    plt.savefig("images/performance.png")
else:
    print("Skipping performance graph - set skip to true if needed")


# In[ ]:
print("Part 3a - comparing approximation and gradient function")

def check_grad(x, y_, epsilon, w1_indices, w0_indices, b0_index, b1_index, W0,
               b0, W1, b1):
    print("PART 3 - check gradient results")
    L0, L1, y = forward(x, W0, b0, W1, b1)
    f_x = cross_entropy(y, y_)

    grads = deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_)

    W1[w1_indices[0]][w1_indices[1]] += epsilon
    copy_L0, copy_L1, copy_out = forward(x, W0, b0, W1, b1)
    f_x_h = cross_entropy(copy_out, y_)

    fin_differences = (f_x_h - f_x) / epsilon
    print("Estimates W1 with finite differences")
    print(fin_differences)
    print("Computed W1:")
    print(grads[0][w1_indices[0]][w1_indices[1]])

    W0, b0, W1, b1 = load_sample_weights()
    W0[w0_indices[0]][w0_indices[1]] += epsilon
    copy_L0, copy_L1, copy_out = forward(x, W0, b0, W1, b1)
    f_x_h = cross_entropy(copy_out, y_)

    fin_differences = (f_x_h - f_x) / epsilon
    print("Estimates W0 with finite differences")
    print(fin_differences)
    print("Computed W0:")
    print(grads[2][w0_indices[0]][w0_indices[1]])


    W0, b0, W1, b1 = load_sample_weights()
    b1[b1_index] += epsilon
    copy_L0, copy_L1, copy_out = forward(x, W0, b0, W1, b1)
    f_x_h = cross_entropy(copy_out, y_)

    fin_differences = (f_x_h - f_x) / epsilon
    print("Estimates b1 with finite differences")
    print(fin_differences)
    print("Computed b1:")
    print(grads[1][b1_index][0])

    W0, b0, W1, b1 = load_sample_weights()
    b0[b0_index] += epsilon
    copy_L0, copy_L1, copy_out = forward(x, W0, b0, W1, b1)
    f_x_h = cross_entropy(copy_out, y_)

    fin_differences = (f_x_h - f_x)/epsilon
    print("Estimates b0 with finite differences")
    print(fin_differences)
    print("Computed b0:")
    print(grads[3][b0_index][0])

W0, b0, W1, b1 = load_sample_weights()
example = M["train2"][0:1].T
true = array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
check_grad(example, true, 0.001, [10,5], [200,10], 5, 7,W0, b0, W1, b1)

