import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('digits_model.keras')
loss, acc = model.evaluate(x_test, y_test)

print('Test loss:', loss)
print('Test accuracy:', acc)