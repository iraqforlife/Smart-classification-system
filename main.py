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
from algos import NeuralNetwork
from algos import SVM
from helpers import datasets as Data
from helpers import utilities as Utils
from sklearn.model_selection import StratifiedShuffleSplit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

####################
#      GLOBAL      #
####################
# Music CSV
JMIRMFCC_CSV = r"music\tagged_feature_sets\msd-jmirmfccs_dev\msd-jmirmfccs_dev.csv"
MARSYAS_CSV = r"music\tagged_feature_sets\msd-marsyas_dev_new\msd-marsyas_dev_new.csv"
RYTHMHISTOGRAM_CSV = r"music\tagged_feature_sets\msd-rh_dev_new\msd-rh_dev_new.csv"

#General config
PRINT_GRAPH = False
BASELINE_CLASSIFIER = ""
#TEST_NBSAMPLES = 50

####################
#       MAIN       #
####################
if __name__ == '__main__':
    #1.A Prepare datasets
    print("PREPARING DATASETS")
    features1, answers1 = Data.prepareTrainDataset("Jmir MFCC", JMIRMFCC_CSV, True)
    features1_train, answers1_train, features1_test, answers1_test = Data.splitDataSet(features1, answers1, 2, 0.8, 0)
    #features2, answers2 = Data.prepareTrainDataset("Marsyas", MARSYAS_CSV, True)
    #features3, answers3 = Data.prepareTrainDataset("Rythm Histogram", RYTHMHISTOGRAM_CSV, True)
    #features, features_SVM, answers, dataset_splitter = Data.prepareDatasetLab2("Galaxy", MERGED_GALAXY_PRIMITIVE, 5, 0.2, 0)

    #2.A Find best algo for specified dataset
    best_data1 = Utils.findBaselineClassifier(['decisionTree', 'bayes', 'knn', 'neuralNetwork'], features1_train, answers1_train)
    #best_data2 = Utils.findBaselineClassifier(['decisionTree', 'knn', 'neuralNetwork', 'SVM'], features2, answers2)
    #best_data3 = Utils.findBaselineClassifier(['decisionTree', 'knn', 'neuralNetwork', 'SVM'], features3, answers3)

    print("END -----------------------------------------------------------")

__authors__ = [
    'Amhad Al-Tahter',
    'Jean-Philippe Decoste',
    'Stéphanie Lacerte'
]
__copyright__ = 'École de technologie supérieure 2018'
