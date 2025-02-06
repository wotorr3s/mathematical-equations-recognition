import tensorflow as tf
import numpy as np

data = tf.keras.utils.image_dataset_from_directory('../data/math_symbols_recognition/extracted_images', batch_size=16, shuffle=True)

images, labels = tuple(zip(*data))