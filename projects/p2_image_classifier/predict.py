# python predict.py ./test_images/orchid.jpg my_model.h5
# python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
# python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import json
import logging

import argparse

with open('label_map.json', 'r') as f:
    class_names = json.load(f)
print(class_names)

filepath = 'best_model.h5'
model = tf.keras.models.load_model(filepath,custom_objects={'KerasLayer':hub.KerasLayer})
reloaded_keras_model.summary()

def predict(image_path, model, top_k,image_size,class_names):
    im = Image.open(image_path)
    image = np.asarray(im)
    ps = model.predict(np.expand_dims(process_image(image,image_size), axis=0))
    top_values, top_indices = tf.math.top_k(ps, top_k)
    top_classes = [class_names[str(value+1)] for value in top_indices.cpu().numpy()[0]]
    return top_values.numpy()[0], top_classes

image_size = 224
probs, classes = predict(image_path, model, top_k, image_size, class_names )

def Sanity_Check(image_path, model, top_k, image_size, class_names):
    probs, classes = predict(image_path0, model, top_k,image_size,class_names)

    im = Image.open(image_path0)
    test_image = np.asarray(im)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,20), ncols=2)
    ax1.imshow(process_image(test_image, image_size), cmap = plt.cm.binary)
    ax1.axis('off')
    ax2.barh(classes[::-1], probs[::-1])
    ax2.set_aspect(0.1)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    return None
