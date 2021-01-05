"""
Sensitivity Analysis for determining the ranking of sensors
"""
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import imutils
import pickle
import keras
import json
import time
import cv2
import csv
import os


# Currently, memory growth needs to be the same across GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
tf.get_logger().setLevel('WARNING')
print(tf.__version__)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the input csv file")
ap.add_argument("-o", "--output", help="path to store graphs")
args = vars(ap.parse_args())

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def test_set_of_sensors(sensor_list):
    num_of_sensors = len(sensor_list)
    df = pd.read_csv(args["input"])
    X = df[sensor_list]
    Y = df['class_label']

    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(X), np.asarray(Y), test_size=0.20, shuffle=True)

    # convert label -1 to 0
    y_train[y_train < 0] = 0
    y_test[y_test < 0] = 0

    # reshape labels data
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # reshape sensor data
    x_train = x_train.reshape(x_train.shape[0], num_of_sensors, 1)
    x_test = x_test.reshape(x_test.shape[0], num_of_sensors, 1)

    model = get_compiled_model(num_of_sensors)
    batch_size = 1
    epochs = 10
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    print(sensor_list, history.history['accuracy'])

def get_compiled_model(num_of_sensors):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_of_sensors, 1), dtype='float32'),
        tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu'),
        tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

test_set_of_sensors(['sensor0', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7', 'sensor8', 'sensor9'])
test_set_of_sensors(['sensor0', 'sensor3', 'sensor4'])
test_set_of_sensors(['sensor5', 'sensor6', 'sensor7', 'sensor8'])
