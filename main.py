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

Group :
GTI770-A18-02
"""

import csv
import math
import os

import graphviz
import numpy as np
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras

from tabulate import tabulate
from helpers import utilities as Utils
from helpers import datasets as Data

####################
#      GLOBAL      #
####################
#LAB01
IMAGE_CSV_NAME = r"data\csv\galaxy\galaxy_label_data_set.csv"
IMAGETEST_CSV_NAME = r"data\csv\galaxy\galaxy_label_data_set_test.csv"

#Galaxy features
EXTRATED_GALAXY_PRIMITIVE = r"data\csv\eq07_pExtraction.csv"
MERGED_GALAXY_PRIMITIVE = r"data\csv\eq07_pMerged.csv"
ALL_GALAXY_PRIMITIVE = r"data\csv\galaxy\galaxy_feature_vectors.csv"

#Spam features
ALL_SPAM_PRIMITIVE = r"data\csv\spam\spam.csv"

#Algo params
TREE_DEPTH = [None, 3, 5, 10]
EXTRACT_TREE_PDF = False
N_NEIGHBORS = [3, 5, 10]
WEIGHTS = ['uniform', 'distance']
LAYERS_ACTIVATION = 'relu'
LAST_LAYER_ACTIVATION = 'sigmoid'
TENSORBOARD_SUMMARY = r"tensorboard"

#General config
PRIMITIVE_SCANNING = False
DOMERGE = False
PRINT_GRAPH = True
#TEST_NBSAMPLES = 50

def neuralNetwork(runId, networkFrame, epoch, learning_rate):
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

    print("1.Initializing Neural Network for run #" + str(runId) + "\n")
    print("TensorFlow version:" + tf.VERSION + ", Keras version:" + tf.keras.__version__ + "\n")

    # Create a default in-process session.
    directory = TENSORBOARD_SUMMARY + "/run" + str(runId)
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("Writing TensorBoard summary writer at :" + directory + "\n")
    tbCallBack = keras.callbacks.TensorBoard(log_dir=directory, histogram_freq=1, write_graph=True, write_images=False)
    
    # Parameters
    dimension = len(features[0])
    layers = networkFrame
    epoch = epoch
    batch_size = 200
    learning_rate = learning_rate
    
    # The network type
    neuralNetwork_model = keras.Sequential()
    counter = 1

    # Set layer in model
    # First layer is set according to data dimension
    neuralNetwork_model.add(keras.layers.Dense(dimension, input_dim=dimension, kernel_initializer='random_normal', bias_initializer='zeros', activation=LAYERS_ACTIVATION))
    neuralNetwork_model.add(keras.layers.Dropout(0.5))
    # Other layer set using layers array
    for perceptron in layers:
        if len(layers) == counter:
            # Last layer (2 neurons for 2 possible class, SIGMOID ensure result between 1 and 0)
            neuralNetwork_model.add(keras.layers.Dense(1, kernel_initializer='random_normal', bias_initializer='zeros', activation=LAST_LAYER_ACTIVATION))
            #print("Layer #" + str(counter) + ": dimension = " + str(2) + ", activation = " + LAST_LAYER_ACTIVATION)
        else:
            # Adds Layer
            neuralNetwork_model.add(keras.layers.Dense(perceptron, kernel_initializer='random_normal', bias_initializer='zeros', activation=LAYERS_ACTIVATION))
            neuralNetwork_model.add(keras.layers.Dropout(0.5))
            #print("Layer #" + str(counter) + ": dimension = " + str(perceptron) + ", activation = " + LAYERS_ACTIVATION)
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
                            verbose=0)

    # Evaluation
    scores = neuralNetwork_model.evaluate(np.array(features_train), np.array(answers_train), verbose=1)
    print("\n%s: %.2f%%" % (neuralNetwork_model.metrics_names[1], scores[1]*100))

    # Clear previous model
    keras.backend.clear_session()

####################
#       MAIN       #
####################
if __name__ == '__main__':
    #0. Generate new feature files for Galaxy (only run once)
    Data.extractFeatures(True, PRIMITIVE_SCANNING, DOMERGE)

    #1.A Read Galaxy features (name of file, path, n_split, test size, random state)
    if os.path.isfile(MERGED_GALAXY_PRIMITIVE):
        features, features_SVM, answers, dataset_splitter = Data.prepareDataset("Galaxy", MERGED_GALAXY_PRIMITIVE, 5, 0.2, 0)
    else:
        features, features_SVM, answers, dataset_splitter = Data.prepareDataset("Galaxy", ALL_GALAXY_PRIMITIVE, 5, 0.2, 0)

    print("ALGORITHMS")
    print("\nNeural Network:")
    neuralNetwork(1, [100, 100, 2], 60, 0.0005)
    neuralNetwork(2, [100, 150, 80, 30, 2], 100, 0.005)
    neuralNetwork(3, [80, 100, 120, 100, 80, 40, 20, 2], 80, 0.05)

    """
    #2.A Execute Algorithm
    print("ALGORITHMS")
    #2.1. DECISION TREE
    print("\nDecision Tree:")
    decisionTree()
    #2.2. KNN
    print("KNN:")
    knn()
    #2.3. BAYES
    print("\nBayes MultinomialNB")
    bayes()




    #3.B Read Spams features (name of file, path, n_split, test size, random state)
    features = []
    answers = []
    features, answers = Data.prepareDataset("Spam", ALL_SPAM_PRIMITIVE, 5, 0.2, 0)

    #4.B Execute Algorithm
    print("ALGORITHMS")
    #4.1. DECISION TREE
    print("\nDecision Tree:")
    decisionTree()
    #4.2. KNN
    print("KNN:")
    knn()
    #4.3. BAYES
    print("\nBayes MultinomialNB")
    bayes()
    """
    
    print("END -----------------------------------------------------------")

__authors__ = [
    'Amhad Al-Tahter',
    'Jean-Philippe Decoste'
]
__copyright__ = 'École de technologie supérieure 2018'
