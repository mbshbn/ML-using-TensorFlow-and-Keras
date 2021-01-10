# python predict.py ./test_images/orchid.jpg my_model.h5
# python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
# python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json


import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import json

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import argparse
#
# creating an ArgumentParser object"
parser = argparse.ArgumentParser(description='Predict name of flower images')
#positional arguments (image_path, model_path):
parser.add_argument('image_path')
parser.add_argument('model_path')
# optional arguments (top_k, category_names):
parser.add_argument('--category_names',
    help ='category_names is json file contains a disctionary maping labels (intigers) to category names (string)',)
parser.add_argument('--top_k',
        help ='top_k is the thenumber of top most likely classes',
        type = int, default =3,  choices=[2, 3, 4, 5, 6, 7])
args = parser.parse_args()

def process_image(image, image_size):
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, (image_size, image_size))
    image/=255
    image= image.numpy()
    return image

im = Image.open(args.image_path)
image = np.asarray(im)
image_size = 224
model = tf.keras.models.load_model(args.model_path,custom_objects={'KerasLayer':hub.KerasLayer})
ps = model.predict(np.expand_dims(process_image(image,image_size), axis=0))
top_values, top_indices = tf.math.top_k(ps, args.top_k)
top_probs = top_values.numpy()[0]

if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    top_classes = [class_names[str(value+1)] for value in top_indices.cpu().numpy()[0]]
    print('\nTop labels are: ', top_indices.numpy())
    print('\nTop probabilities are: ',top_probs)
    print('\nTop classess are: ',top_classes)
else:
    print('\nTop labels are: ', top_indices.numpy())
    print('\nTop probabilities are: ',top_probs)
