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

#matplotlib inline  

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
#imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
#show()


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output
    
def cross_entropy(y, y_):
    return -sum(y_*log(y)) 

def NLL(y, y_):
    return -sum(y_*log(y)) 

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network - NOT WORKING'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T ) 
    dCdobydodh = dot(W1, dCdL1)
    one_minus_h_sq = 1-L0**2

    dCdW0 = tile(dCdobydodh, 28*28).T * dot(x, (one_minus_h_sq.T))
    dCdb1 = dCdL1
    dCdb0 = dCdobydodh * one_minus_h_sq
    
    return dCdW1, dCdb1, dCdW0, dCdb0


'''
    Display images of the training set, showing 10 images per digit 0 to 9
    and save as "p1_results.png"
'''
def part1():
    
    f, sub = plt.subplots(10, 10,figsize=(15,15))
    for i in range(10):
        print("Getting images for digit {:d}".format(i))
        for j in range(10):
            sub[i][j].axis('off')
            sub[i][j].imshow(M['train' + str(i)][j+20].reshape((28,28)), cmap=cm.gray)
            print("\timage{:d}".format(j))
    
    print("Saved images to file: p1_results.png")
    savefig('p1_results.png')



'''
    A simple neural network that computes outputs o_i 
    with o_i = su_j w_ji x_j + b_i.
    
    Some of this code is from Guerzhoy:
    http://www.cs.toronto.edu/~guerzhoy/411/proj2/mnist_handout.py
    
'''
def part2():
    print("===================================================================")
    print("running part 2")
    print()
    img = "train5"
    
    #Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("snapshot50.pkl","rb"), encoding="latin1")
    
    # works up to digits 1-3, not 4-9     # 300 nodes
    
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1)) # 10 biases for 10 digits(0-9)
    
    correct = 0
    ## test performance of this network on all digits(0-9)
    for i in range(0, 10):
    
        # Load one example from the training set, and run it through the
        # neural network to predict what digit it is
        img = 'train' + str(i)
        x = M[img][148:149].T    
        output = dot(x, w0) + b0
        
        # get the index at which the output is the largest, the most likely digit
        # found in this image
        y = argmax(output)
        
        print("actual digit: {:s}".format(img[-1]))
        print("predicted digit: {:d}".format(y))
        
        if int(img[-1]) == y:
            print ("CORRECT")
            correct += 1
        print()
        # y_true = array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
    
    print("ACCURACY: {:f}".format(correct/10))
    print("===================================================================")
    

''' 
    PART 3A,
    A negative log likelihood cost function
    y is a n * m vector of targets/labels
        m = # of cases(images)
        n = number of inputs(pixels per image)
    y_hat are predicted targets, the one-hot encoded probabilities we get from 
    the NN.
    the p_i's from slide 7 of the one-hot encoding slides on Guerzhoy's site
'''
def C(y, outputs):
    return -sum(y * log(y_hat))
    

'''
    PART 3A
    Compute gradient of the negative log cost function C()
    
    We know that:
    dC/dw = dC/do * do/dw
'''
def dCdw(x, y, y_hat, w, b):
    dCdo = y_hat - y
    
   
    # append x_o = 1 to x vector
    temp = vstack((ones((1, x.shape[1])), x))))
    
    # o_i = x_j + b_i    from P2 handout in Part 2 description
    # so dodw = x_j + b_i
    dodw = temp
    
    return dot(dCdo.T, dodw)


'''
    Return one-hot encoding target for a specified digit from 0 to 9.
    
    Example: If the digit in a picture is 9, then it's one-host encoding label
    this function will return is this:
    
    > get_one_hot_code(9)
    
    [0 0 0 0 0 0 0 0 0 1]
'''
def get_one_hot_code(i)
    y = zeros((10, 1))
    y[i] = 1
    return y

def single_forward(x, w, b):
    return dot(x, w) + b

'''
     Verify that the gradient of the cost function, dCdw() is correct with finite
     difference approximation.
'''
def part3b():
    # TODO: plot learning curve of training set with x pics per digit versus performance, classifier accuracy
    
    for i in range(0, 10):
        h = 1e-5                    # finite difference
        
        x = random.randint(0, 255, (32*32, 1)) / 255.0
        y = zeros((10, 1))
        y[i] = h
        
        w = zeros((784, 10))         # for 32 * 32 = 784 pixels, each can be 1 of 10 digits 0 to 9
        b = zeros((10, 1))           # there are 0 - 9 possible digits
        
        output = single_forward(x, w, b)
        gradient = dCdw(x, t, y, y_hat, w, b)
        
        # calculate the gradient at different points
        y_1 = single_forward(x, w, b + get_one_hot_code(i))
        y_2 = single_forward(x, w, b - get_one_hot_code(i))
        
        # determine the finite difference
        finite_diff = (o_1 - o_2) / (2 * h)
        
        # compare the finite diff with gradient
        print("estimated gradient: "format(gradient(i)))
        print("actual".format(gradient[0][i]))



# 
# 
# def part4():
#     print("===================================================================")
#     print("running part 2")
#     print()
#     img = "train5"
#     
#     #Load sample weights for the multilayer neural network
#     snapshot = cPickle.load(open("snapshot50.pkl","rb"), encoding="latin1")
#     
#     # works up to digits 1-3, not 4-9     # 300 nodes
#     
#     W0 = snapshot["W0"]
#     b0 = snapshot["b0"].reshape((300,1))
#     W1 = snapshot["W1"]
#     b1 = snapshot["b1"].reshape((10,1)) # 10 biases for 10 digits(0-9)
#     
#     correct = 0
#     ## test performance of this network on all digits(0-9)
#     for i in range(0, 10):
#     
#         # Load one example from the training set, and run it through the
#         # neural network to predict what digit it is
#         img = 'train' + str(i)
#         x = M[img][148:149].T    
#         L0, L1, output = forward(x, W0, b0, W1, b1)
#         
#         # get the index at which the output is the largest, the most likely digit
#         # found in this image
#         y = argmax(output)
#         
#         print("actual digit: {:s}".format(img[-1]))
#         print("predicted digit: {:d}".format(y))
#         
#         if int(img[-1]) == y:
#             print ("CORRECT")
#             correct += 1
#         print()
#         # y_true = array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T
#     
#     print("ACCURACY: {:f}".format(correct/10))
#     print("===================================================================")
#     ################################################################################
#     #Code for displaying a feature from the weight matrix mW
#     #fig = figure(1)
#     #ax = fig.gca()    
#     #heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#     #fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#     #show()
#     ################################################################################