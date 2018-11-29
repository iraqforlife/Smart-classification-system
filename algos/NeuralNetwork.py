#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab 1 - Lab's name

Students :
Amhad Al-Taher — Permanent Code
Jean-Philippe Decoste - DECJ19059105
Stéphanie Lacerte - LACS

Group :
GTI770-A18-02
"""

from helpers import utilities as Utils
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tabulate import tabulate

LAYERS_ACTIVATION = 'relu'
LAST_LAYER_ACTIVATION = 'sigmoid'
TENSORBOARD_SUMMARY = r"tensorboard"

def neuralNetwork(runId, networkFrame, epoch, learning_rate, features, answers, dataset_splitter):
    # Format arrays to np arrays
    features_train = []
    answers_train = []
    features_test = []
    answers_test = []

    for train_index, test_index in dataset_splitter.split(features, answers):
        for elem in train_index:
            features_train.append(features[elem])
            answers_train.append(answers[elem])

        for elem in test_index:
            features_test.append(features[elem])
            answers_test.append(answers[elem])

    print("1.Initializing Neural Network for run #" + str(runId))

    # Create a default in-process session.
    directory = TENSORBOARD_SUMMARY + "/run" + str(runId)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("TensorBoard summary writer at :" + directory + "\n")
    tbCallBack = keras.callbacks.TensorBoard(log_dir=directory, histogram_freq=0, write_graph=False, write_images=False)
    
    # Parameters
    dimension = len(features[0])
    layers = networkFrame
    epoch = epoch
    batch_size = 600
    learning_rate = learning_rate
    
    # The network type
    neuralNetwork_model = keras.Sequential()
    counter = 1

    # Set layer in model
    # First layer is set according to data dimension
    neuralNetwork_model.add(keras.layers.Dense(dimension, input_dim=dimension, kernel_initializer='random_normal', bias_initializer='zeros', activation=LAYERS_ACTIVATION))
    neuralNetwork_model.add(keras.layers.Dropout(0.2))
    # Other layer set using layers array
    for perceptron in layers:
        if len(layers) == counter:
            # Last layer (2 neurons for 2 possible class, SIGMOID ensure result between 1 and 0)
            neuralNetwork_model.add(keras.layers.Dense(1, kernel_initializer='random_normal', bias_initializer='zeros', activation=LAST_LAYER_ACTIVATION))
        else:
            # Adds Layer
            neuralNetwork_model.add(keras.layers.Dense(perceptron, kernel_initializer='random_normal', bias_initializer='zeros', activation=LAYERS_ACTIVATION))
            neuralNetwork_model.add(keras.layers.Dropout(0.2))
        counter = counter + 1

    # Compile the network according to previous settings
    neuralNetwork_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate), 
                                loss='binary_crossentropy', 
                                metrics=['accuracy'])

    # Print visualisation of network (layer and perceptron)
    neuralNetwork_model.summary()

    # Fit model to data
    print("\n2.Training\n")
    neuralNetwork_model.fit(np.array(features_train), np.array(answers_train),
                            epochs=epoch,
                            batch_size=batch_size,
                            validation_data=(np.array(features_test), np.array(answers_test)),
                            callbacks=[tbCallBack],
                            verbose=2)

    # Evaluation
    #scores = neuralNetwork_model.evaluate(np.array(features_train), np.array(answers_train), verbose=1)
    #print("\n%s: %.2f%%" % (neuralNetwork_model.metrics_names[1], scores[1]*100))

    # Clear previous model
    keras.backend.clear_session()
