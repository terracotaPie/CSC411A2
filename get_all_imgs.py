from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
# from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import toimage
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
from scipy.ndimage import imread
import urllib
from urllib import *
import pickle as cPickle
import urllib.request


from PIL import Image

'''
    get images of actors in 
 act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    and put in 28 x 28 cropped grayscale images in a folder:
    

'''

#act = list(set([a.split('\t')[0] for a in open(input_file).readlines()]))
act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# input_file = 'facescrub_actors.txt'
#input_file = 'facescrub_actresses.txt'
input_files = [os.getcwd() + '/img_data/facescrub_actors.txt',\
               os.getcwd() + '/img_data/facescrub_actresses.txt']

IMG_SIZE = (28, 28)

cropped_dir = os.getcwd() + '/' + 'img_data/cropped/'
uncropped_dir = os.getcwd() + '/' + 'img_data/uncropped/'

# create folders 'uncropped' and 'cropped' for script to work
if not os.path.exists(uncropped_dir):
    os.makedirs(uncropped_dir)
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

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

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result



# testfile = urllib.URLopener()       # for python 2.7           
testfile = urllib.request.URLopener()         # for python 3.5

cropped_imgs = {}

for input_file in input_files:

    # format is x1,y1,x2,y2, 
    # top left then bottom right corner point
    
    for a in act:
        cropped_imgs[a] = []
        name = a.split()[1].lower()
        i = 0
        for line in open(input_file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], 'uncropped/'+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], uncropped_dir + '/'+filename), {}, 30)
                if not os.path.isfile(uncropped_dir + '/'+filename):
                    continue
    
                
                print(filename)
                
    ## ADDED CODE BELOW
                bbox = [int(x) for x in line.split()[5].split(',')]
                x1 = bbox[0]       # top-left corner
                y1 = bbox[1]
                
                x2 = bbox[2]       # bottom-right corner
                y2 = bbox[3]
        
                # find face bounding box, then crop, grayscale, resize and save
                try:
                    img = imread(uncropped_dir + '/' + filename)
                except IOError:
                    continue
                
                #read in the downloaded image
                img = imread(uncropped_dir + '/' + filename, True)
                print( line.split()[4].split('/')[-1], ' --> ', filename)
                print( 'original shape', img.shape)
                print( 'bounding box:')
                print( '    y: {} to {}'.format(y1, y2))
                print( '    x: {} to {}'.format(x1, x2))
            
                
                ## crop out face and, convert img to grayscale
                
                # check if image is smaller than bounding box and resize image accordingly.
                cropped = img.copy()
    
                # bounding box height is larger than image
                if y2 >= img.shape[0] and y2 >= x2:
                    print( 'height bigger')
                    proportion = ceil(double(y2)/img.shape[0])
                    print( 'proportion', proportion)
                    cropped = imresize(cropped, proportion, 'bilinear')
                
                # bounding box width is larger than image
                elif x2 >= img.shape[1] and x2 >= y2:
                    print( 'width bigger')
                    proportion = ceil(double(x2)/img.shape[1])
                    print( 'proportion', proportion)
                    cropped = imresize(cropped, proportion, 'bilinear')
    
                # crop the actor/actress's face out
                cropped = cropped[y1:y2, x1:x2]
                
                #resize the image to 32 x 32 and save in the folder 'cropped'
                cropped = imresize(cropped, IMG_SIZE, 'bilinear')    # call scipy.misc
                #cropped = toimage(cropped)
                
                newFile = filename.replace('.jpg', '-cropped.jpg')
                
                # convert to PIL image type to save and retain grayscaling
                # save the whole cropped images to the cropped_dir folder
                # toimage(cropped).save(cropped_dir + '/' + newFile)
                
                cropped_imgs[a].append((filename, toimage(cropped)))

    ## end of added code
                i += 1
pf = open(os.getcwd() + '/img_data/actor_imgs.p', 'wb')
cPickle.dump(cropped_imgs, pf)
pf.close()


# to read
# f = open(os.getcwd() + '/img_data/actor_imgs.p', 'rb')
# loaded_imgs = cPickle.load(f)
# f.close()