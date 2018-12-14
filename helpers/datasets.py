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

import csv
import math
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import decomposition
from helpers import utilities as Utils
from helpers import music

LABEL_ENCODER = None

def organizeFeatures(row, isTagged):
    """
    Organise les features. Assume que le fichier est de forme suivante:
    [0 = ID de la ligne, 1 = ID de la chanson, dernière = Classe de la chanson]
    Args:
        row:        Ligne du fichier CSV
        isTagged:   Le fichier contient-il la classe?
    """
    # Save answer
    answer = []
    if isTagged:
        answer = row[-1]
        # Remove from file
        del row[-1]

    # Save music ID
    musicId = row[1]    
    # Remove column ID and track ID
    del row[0] # The identifier (ex:27)
    del row[0] # The id (ex:TRAAAAK128F9318786)

    # Save all features left in row
    features = [float(i) for i in row]

    if isTagged:
        return features, answer
    else:
        return features, musicId

def prepareTrainDataset(datasetName, dataset, isTraining):
    """
    Prépare le dataset pour les algos. 1. Organisation, 2. Normalisation, 3. Réduction de dimension
    Args:
        datasetName:    Nom du CSV
        dataset:        Chemin vers le CSV
        isTraining:     Fichier d'entrainement?
    """
    allData_length = len(list(csv.reader(open(dataset))))
    progress = 0

    features = []
    answers = []
    ids = []

    print("Reading " + datasetName + " features:")
    Utils.printProgressBar(0, allData_length, prefix='Progress:', suffix='Complete', length=50)
    with open(dataset, 'r') as theFile:
        lines = csv.reader(theFile, delimiter=',', quotechar='|')

        for row in lines:
            progress += 1
            Utils.printProgressBar(progress+1, allData_length, prefix='Progress', suffix='Complete', length=50)

            if isTraining:
                feature, answer = organizeFeatures(row, True)
                answers.append(answer)
            else:
                feature, mId = organizeFeatures(row, True)
                ids.append(mId)
            features.append(feature)

    #4. Scale data
    print("\nScaling data...")
    dataScaler = MinMaxScaler(feature_range=(0, 1))
    dataScaler.fit(features)
    features_scaled = dataScaler.transform(features)

    #5 PCA
    print("Reducing dimension...")
    features_PCA = decomposition.PCA(n_components=0.95).fit_transform(features_scaled)

    if isTraining:
        print("Encoding labels...")
        #6 Transform labels
        le = LabelEncoder()
        le.fit(answers)
        answers = le.transform(answers)
        LABEL_ENCODER = le

    print("\n-> Done\n")

    if isTraining:
        return features_PCA, answers, le
    else:
        return features_PCA, ids
    
def splitDataSet(features, answers, n_splits, test_size, random_state):
    
    #Split dataset using model_selection
    print("Splitting Dataset according to these params:")
    print(tabulate([['Property', 'Value'], ['n_splits', n_splits], ['test_size', test_size], ['random_state', random_state]], headers="firstrow"))
    
    dataset_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    # Format arrays to np arrays
    features_train = []
    answers_train = []
    features_test = []
    answers_test = []
    splited = dataset_splitter.split(features, answers)
    
    trainIndexes = []
    testIndexes = []
    
    for train_index, test_index in splited:
        trainIndexes = train_index 
        testIndexes = test_index
        features_train, features_test = features[train_index], features[test_index]
        answers_train, answers_test = answers[train_index], answers[test_index]
    
    print("-> Done\n\n")

    return features_train, answers_train, features_test, answers_test, trainIndexes, testIndexes

def outPut(ids, answers):
    assert len(ids) == len(answers)
    file = open("output.csv", "w")

    for index in range(0, len(ids)):
        file.write("%s;%s" % (ids[index], answers[index]))
        file.write("\n")
    print("file creation is done.")
