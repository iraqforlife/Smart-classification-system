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

from algos import decisionTree as Tree
from algos import Knn as Knn
from algos import Bayes as Bayes
import csv
import math
import os

import graphviz
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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
LAYERS_ACTIVATION = 'relu'
LAST_LAYER_ACTIVATION = 'sigmoid'
TENSORBOARD_SUMMARY = r"tensorboard"

#General config
PRIMITIVE_SCANNING = False
DOMERGE = False
PRINT_GRAPH = True
#TEST_NBSAMPLES = 50

def svm():    
    #linear    
    print("SVM linear")
    c=[0.001,0.1,1.0,10.0]
    params = dict(kernel=['linear'], C=c ,class_weight=['balanced'], cache_size=[2048])
    grid = GridSearchCV(SVC(), param_grid=params, cv=dataset_splitter, n_jobs=-1, iid=True)
    
    #Fit the feature to svm algo
    grid.fit(features_SVM, answers)
    
    #build table
    outPut = []
    y1 = []
    for i in range(0, 4):
        outPut.append([grid.cv_results_['params'][i]['C'],
                          "{0:.2f}%".format(grid.cv_results_['mean_test_score'][i]*100)])
        y1.append(grid.cv_results_['mean_test_score'][i]*100)
    
    #print table
    print(tabulate(outPut, headers=['Variable C','class_weight= {‘balanced’}']))
    print("The best parameters are ", grid.best_params_," with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
    
    #rbf
    print("\nSVM rbf")
    params = dict(kernel=['rbf'], C=c, gamma=c ,class_weight=['balanced'], cache_size=[2048])
    grid = GridSearchCV(SVC(), param_grid=params, cv=dataset_splitter, n_jobs=-1, iid=True)
    
    #Fit the feature to svm algo
    grid.fit(features_SVM, answers)
    
    #build table
    outPut = []
    outPut.append(["0.001",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][0]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][1]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][2]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][3]*100)])
    outPut.append(["0.1",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][4]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][5]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][6]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][7]*100)])
    outPut.append(["1.0",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][8]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][9]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][10]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][11]*100)])
    outPut.append(["10.0",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][12]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][13]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][14]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][15]*100)])
    
    #print table
    print(tabulate(outPut, headers=['Variable C','a=0.001','a=0.1','a=1.0','a=10.0']))
    print("The best parameters are ", grid.best_params_," with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
    
    print("-> Done\n\n")

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

####################
#       MAIN       #
####################
if __name__ == '__main__':
    #0. Generate new feature files for Galaxy (only run once)
    Data.extractFeatures(True, PRIMITIVE_SCANNING, DOMERGE)

    #1.A Read Galaxy features (name of file, path, n_split, test size, random state)
    if os.path.isfile(MERGED_GALAXY_PRIMITIVE):
        features, features_SVM, answers, dataset_splitter = Data.prepareDataset("Galaxy", MERGED_GALAXY_PRIMITIVE, 5, 0.3, 0)
    else:
        features, features_SVM, answers, dataset_splitter = Data.prepareDataset("Galaxy", ALL_GALAXY_PRIMITIVE, 5, 0.3, 0)

    print("ALGORITHMS")
    print("\nDecision Tree:")
    Tree.decisionTree(features, answers)
    print("\nSVM:")
    #svm()

    print("\nNeural Network:")
    print("TensorFlow version:" + tf.VERSION + ", Keras version:" + tf.keras.__version__ + "\n")
    # Diff number of layer
    #neuralNetwork(1, [100, 100, 2], 60, 0.0005)
    #neuralNetwork(2, [100, 2], 60, 0.0005)
    #neuralNetwork(3, [100, 100, 100, 100, 2], 60, 0.0005)
    # Diff perceptron
    #neuralNetwork(4, [80, 50, 2], 60, 0.0005)
    #neuralNetwork(5, [120, 2], 60, 0.0005)
    #neuralNetwork(6, [100, 120, 100, 50, 2], 60, 0.0005)
    # Diff epoch
    #neuralNetwork(7, [100, 100, 2], 60, 0.0005)
    #neuralNetwork(8, [100, 2], 20, 0.0005)
    #neuralNetwork(9, [100, 100, 100, 100, 2], 100, 0.0005)
    # Diff learning
    #neuralNetwork(10, [100, 100, 2], 60, 0.0005)
    #neuralNetwork(11, [100, 2], 60, 0.005)
    #neuralNetwork(12, [100, 100, 100, 100, 2], 60, 0.05)

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
