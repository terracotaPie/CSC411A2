# File Listing
What's for which?


## PART 1-5
### digits.py
classify digits with NN


## PART 7-9 FACE CLASSIFICATION

### faces.py
classify imgs apart from 6 actors using a single hidden-layer NN with TensorFlow

### get_all_imgs.py
Gather actors' images in to train in NN in PART 7
This crops images, scales them to 28 x 28 images and greyscales them to save in a pickle file `actor_imgs.p`.

### actor_imgs.p
This pickle file will store actors' images in a dictionary of the format:
```
actor_imgs = {
	'actor name': [('filename', <PIL IMG OBJECT>), ('filename2', img2), ...]
	'Bill Hader': [('hader0.jpg', img4), ('hader3.jpg', img5), ...]
}
```
### Related Folder:
```
img_data/tf/
	-> cropped/                        # 28 * 28 grey imgs
	-> uncropped/					   # uncropped RGB imgs
	-> facescrub_actors.txt            # list of actorsto download
	-> facescrub_actresses.txt         # list of actresses
	-> actor_imgs.p                    # pickle file of cropped actor images

```


## Part 10-11 ALEX CONVNET

### bvlc_alexnet.py

### caffe_classes.py

### myalexnet_forward_newtf.py

### Associated Folder
```
img_data/alexnet/
	imgs/              # contains NN testing imgs
	
```