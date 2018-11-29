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

from algos import DecisionTree
from algos import Knn
from algos import Bayes
import csv
import math
import os

import graphviz
import numpy as np
from sklearn.model_selection import GridSearchCV

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


#General config
PRINT_GRAPH = False
#TEST_NBSAMPLES = 50

####################
#       MAIN       #
####################
if __name__ == '__main__':
    #1.A Read features (name of file, path, n_split, test size, random state)
    if os.path.isfile(MERGED_GALAXY_PRIMITIVE):
        features, features_SVM, answers, dataset_splitter = Data.prepareDataset("Galaxy", MERGED_GALAXY_PRIMITIVE, 5, 0.3, 0)
    else:
        features, features_SVM, answers, dataset_splitter = Data.prepareDataset("Galaxy", ALL_GALAXY_PRIMITIVE, 5, 0.3, 0)

    print("ALGORITHMS")
    print("\nDecision Tree:")
    #DecisionTree.decisionTree(features_SVM, answers, PRINT_GRAPH)
    print("\nSVM:")
    Knn.knn(features, answers, PRINT_GRAPH)
    #svm()

    print("\nNeural Network:")
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
