"""
#Deep Dreaming in Keras.
Run the script with:
```python
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```
e.g.:
```python
python deep_dream.py img/mypic.jpg results/dream
```
"""

from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy
import argparse

from keras_applications import inception_v3
from keras import backend as K


parser = argparse.ArgumentParser(description='Deep Dreams with Keras(Inception Network).')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# These are the names of the layers
# for which we try to maximize activation,
# as well as their weight in the final loss
# we try to maximize.
# You can tweak these setting to obtain new visual effects.
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}

# utility function resize, format inout image into appropriate tensors
def preprocess_image(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = inception_v3.preprocess_input(image)
    return image

# utility function converts tensor into valid image
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1,2,0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))

    x /= 2.
    x += 0.5
    x *= 255
    image = np.clip(x,0,255).astype('uint8')
    return image