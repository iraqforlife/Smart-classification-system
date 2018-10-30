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
import matplotlib.pyplot as plt
import numpy as np
from image import Image as imageObj
from imageV2 import Image as imageFeat
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from spam import Spam
from tabulate import tabulate


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

#General config
PRIMITIVE_SCANNING = True
DOMERGE = False
PRINT_GRAPH = True
#TEST_NBSAMPLES = 50




####################
#       LAB01      #
####################
def lab1_extractFeatures():
    destFile_rowCount = 0
    sourceFile_rowCount = 0
    allGalaxy_length = len(list(csv.reader(open(ALL_GALAXY_PRIMITIVE))))

    #If source images csv row count > extracted features csv row count, 
    # we process each images in the file
    if os.path.isfile(EXTRATED_GALAXY_PRIMITIVE):
        destFile_rowCount = len(list(csv.reader(open(EXTRATED_GALAXY_PRIMITIVE))))
        sourceFile_rowCount = len(list(csv.reader(open(IMAGE_CSV_NAME)))) - 1

    if PRIMITIVE_SCANNING and sourceFile_rowCount > destFile_rowCount:
        #imgDataset = loadAllImages(IMAGETEST_CSV_NAME, r"data\images") #test
        imgDataset = loadAllImages(IMAGE_CSV_NAME, r"data\images") #prod

        with open(EXTRATED_GALAXY_PRIMITIVE, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(imgDataset)

    #Merge feature we extracted
    if DOMERGE:
        print("MERGE")
        valueFirstFile = []
        mergedFiles = []
        progress = 0

        print("Reading file '" + EXTRATED_GALAXY_PRIMITIVE + "'")
        printProgressBar(0, destFile_rowCount, prefix='Progress:', suffix='Complete', length=50)
        with open(EXTRATED_GALAXY_PRIMITIVE, "r") as newFeature:
            reader = csv.reader(newFeature, delimiter=',', quotechar='|')
            for row in reader:
                progress += 1
                printProgressBar(progress+1, destFile_rowCount, prefix='Progress', suffix='Complete', length=50)

                values = [float(i) for i in row]
                valueFirstFile.append(values)
        print("\n-> Done\n")

        progress = 0
        print("Reading file '" + ALL_GALAXY_PRIMITIVE + "' and merging it with the first one")
        printProgressBar(0, allGalaxy_length, prefix='Progress:', suffix='Complete', length=50)
        with open(ALL_GALAXY_PRIMITIVE, "r") as allFeature:
            reader = csv.reader(allFeature, delimiter=',', quotechar='|')
            for row in reader:
                progress += 1
                printProgressBar(progress+1, allGalaxy_length, prefix='Progress', suffix='Complete', length=50)

                for firstFileRow in valueFirstFile:
                    if int(firstFileRow[0]) == int(math.ceil(float(row[0]))):
                        values = []
                        lastColumn = row[-1:]

                        values.append(int(math.ceil(float(row[0]))))
                        for i in range(1, len(row) - 1):
                            values.append(float(row[i]))

                        for i in range(1, len(firstFileRow)):
                            values.append(float(firstFileRow[i]))

                        values.append(int(math.ceil(float(lastColumn[0]))))
                        mergedFiles.append(values)
        print("\n-> Done\n")

        print("Writing the new file '" + MERGED_GALAXY_PRIMITIVE + "'")
        with open(MERGED_GALAXY_PRIMITIVE, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(mergedFiles)
        print("-> Done\n\n")

def loadAllImages(dataPath, imageFolderPath):
    """
    This method is used to al the images from the data set.

    Args:
        dataPath: the dataset file path
        imageFolderPath: the image dataset folder path
    """
    #Opened CSV file (r = read)
    dataFile = open(dataPath, 'r')
    #Skip header
    next(dataFile)

    tempDataset = []
    fileLength = len(list(csv.reader(open(dataPath))))
    # Initial call to print 0% progress
    progress = 0
    print("Extracting our own features:")
    printProgressBar(0, fileLength, prefix='Progress:', suffix='Complete', length=50)
    for line in dataFile:
        #Prepare data
        texts = line.split(",")
        imgName = texts[0]
        shape = str(texts[1])
        imgPath = imageFolderPath + "\\" + str(imgName) + '.jpg'

        #Print progress
        progress += 1
        #sleep(0.1)
        printProgressBar(progress + 1, fileLength, prefix='Progress:', suffix='Complete', length=50)

        #The image
        img = imageObj(imgPath, imgName, shape)

        #Push data
        tempDataset.append([img.Label, img.BlackWhiteRatio, img.C, img.Convexity, img.B])

    dataFile.close()

    return tempDataset

def splitDataset(og_dataset, shuffle, option):
    """
    This method is used to al the images from the data set.

    Args:
        og_dataset: the dataset
        shuffle: do shuffle before split
        option: how to split the file in 2
    
    limit = len(og_dataset)

    if shuffle:
        random.shuffle(og_dataset)

    if option == "test":
        limit = TEST_NBSAMPLES
    
    for i in range(0, limit):
        if i % 3 != 2:
            trainDataset.append(og_dataset[i])
        else:
            testDataset.append(og_dataset[i])
    """

def traceGraph(feature1x, feature1y, feature1Name, feature2x, feature2y, feature2Name, xlabel, ylabel):
    """
    This method is used to out a 2D graph with the selected features

    Args:
        feature1x : The feature 1 x array. It will be used for the first data set shown in the graph. 
        feature1y : The feature 1 y array. It will be used for the first data set shown in the graph.
        feature1Name : The feature 1 dataset name. It will be used for the first data set shown in the graph.
        feature2x : The feature 2 x array. It will be used for the first data set shown in the graph.
        feature2y : The feature 2yx array. It will be used for the first data set shown in the graph.
        feature2Name : The feature 2 dataset name. It will be used for the first data set shown in the graph.
        xlabel : the label shown on the x axis.
        ylabel: the label shown on the y axis:
     """
    #fig = plt.figure()
    #ax1 = fig.add_subplot()
    plt.grid(True)
    plt.scatter(feature1x, feature1y, s=10, c='b', marker="s", label=feature1Name)
    plt.scatter(feature2x, feature2y, s=10, c='r', marker="o", label=feature2Name)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.legend(loc='upper left')
    plt.show()

####################
#       LAB02      #
####################
def lab2_prepareDataset(datasetName, dataset, n_splits, test_size, random_state):
    print("PREPARING DATASETS")
    allData_length = len(list(csv.reader(open(dataset))))
    progress = 0
    datas = []

    print("Reading " + datasetName + " features:")
    printProgressBar(0, allData_length, prefix='Progress:', suffix='Complete', length=50)
    with open(dataset, 'r') as theFile:
        primitives = csv.reader(theFile, delimiter=',', quotechar='|')

        for row in primitives:
            progress += 1
            printProgressBar(progress+1, allData_length, prefix='Progress', suffix='Complete', length=50)

            values = [float(i) for i in row]
            if datasetName == "Galaxy":
                datas.append(imageFeat(values))
            elif datasetName == "Spam":
                datas.append(Spam(values))
    print("\n-> Done\n")

    #3. Split dataset using model_selection
    print("Splitting Dataset according to these params:")
    print(tabulate([['Property', 'Value'], ['n_splits', n_splits], ['test_size', test_size], ['random_state', random_state]], headers="firstrow"))

    features = []
    answers = []
    for data in np.array(datas):
        features.append(data.features)
        answers.append(data.answer)
    dataset_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    for train_index, test_index in dataset_splitter.split(features, answers):
        for elem in train_index:
            features_train.append(features[elem])
            answers_train.append(answers[elem])

        for elem in test_index:
            features_test.append(features[elem])
            answers_test.append(answers[elem])
    print("-> Done\n\n")

#### ALGORITHMS ####
def decisionTree():
    """
    This method is used to build the decision tree and how the graph associated with it.
    fit(x,y) takes two arrays:
    X, sparse or dense, of size [n_samples, n_features] holding the training samples, and an array
            ex: [[0, 0], [1, 1]]
    Y of integer values, size [n_samples], holding the class labels
            ex : [0, 1]

    Since out Y array is not numerical we are using preprocessing from klearn to transform them
    Args:
        data: the array containing all the features. It will be used as the X
        labels: the array containing all the labels. It will be used as the Y
    """
    print("1.Training\n")
    dTreePerf = [['Depth', 'Score']]
    params = dict(max_depth=TREE_DEPTH)
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5, n_jobs=-1, verbose=50)

    #Fit data to Decision Tree algo
    grid.fit(features_train, answers_train)

    #Loop through results
    for i in range(0, 4):
        dTreePerf.append([grid.cv_results_['params'][i]['max_depth'],
                        "{0:.2f}".format(grid.cv_results_['mean_test_score'][i]*100)])

    print(tabulate(dTreePerf, headers="firstrow"))
    print("\nThe best is depth = %s" %(grid.best_params_['max_depth']))

    print("\n2.Validating performance with test data\n")
    answer_true, answer_pred = answers_test, grid.predict(features_test)
    print(classification_report(answer_true, answer_pred))

def knn():
    print("1.Training\n")
    knnPerf = [['Weights', 'K', 'Score']]
    params = dict(n_neighbors=N_NEIGHBORS, weights=WEIGHTS, algorithm=['auto'])
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=5, n_jobs=-1)

    #Fit data to knn algo
    grid.fit(features_train, answers_train)

    #Loop through results
    for i in range(0, 6):
        knnPerf.append([grid.cv_results_['params'][i]['weights'],
                        grid.cv_results_['params'][i]['n_neighbors'],
                        "{0:.2f}".format(grid.cv_results_['mean_test_score'][i]*100)])

    print(tabulate(knnPerf, headers="firstrow"))
    print("\nThe best is KNN %s With K = %s" %(grid.best_params_['weights'], grid.best_params_['n_neighbors']))

    print("\n2.Validating performance with test data\n")
    answer_true, answer_pred = answers_test, grid.predict(features_test)
    print(classification_report(answer_true, answer_pred))

def bayes():
    print("1.Training\n")
    bayesPerf = [['Data type', 'Score']]
    params = dict()
    grid = GridSearchCV(MultinomialNB(), param_grid=params, cv=5, n_jobs=-1)

    #Scale the data between 0 and 1
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(features_train)
    xNormalized = scaler.transform(features_train)

    #Fit normalized data to Bayes algo
    grid.fit(xNormalized, answers_train)
    bayesPerf.append(['Normalized', "{0:.2f}".format(grid.best_score_*100)])
    answer_true, answer_pred = answers_test, grid.predict(features_test)
    
    #Discretize normalized data
    est = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    est.fit(features_train)
    xKBinsDiscretizer = est.transform(features_train)

    #Fit discretize data to bayes algorithm
    grid.fit(xKBinsDiscretizer, answers_train)
    bayesPerf.append(['Discretized', "{0:.2f}".format(grid.best_score_*100)])

    print(tabulate(bayesPerf, headers="firstrow"))

    print("\n2.Validating performance with test data\n")
    
    print(classification_report(answer_true, answer_pred))

####################
#    UTILITIES     #
####################
def printProgressBar(iteration, total, prefix='', suffix='', length=50, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')

def printGraph():
    #fig = plt.figure()
    #ax1 = fig.add_subplot()
    plt.grid(True)
    #plt.scatter(feature1x, feature1y, s=10, c='b', marker="s", label=feature1Name)
    #plt.scatter(feature2x, feature2y, s=10, c='r', marker="o", label=feature2Name)
    #plt.ylabel(xlabel)
    #plt.xlabel(ylabel)
    #plt.legend(loc='upper left')
    #plt.show()

####################
#       MAIN       #
####################
if __name__ == '__main__':
    #0. Generate new feature files for Galaxy (only run once)
    lab1_extractFeatures()

    #1.A Read Galaxy features (name of file, path, n_split, test size, random state)
    features_train = []
    features_test = []
    answers_train = []
    answers_test = []
    if os.path.isfile(MERGED_GALAXY_PRIMITIVE):
        lab2_prepareDataset("Galaxy", MERGED_GALAXY_PRIMITIVE, 5, 0.2, 0)
    else:
        lab2_prepareDataset("Galaxy", ALL_GALAXY_PRIMITIVE, 5, 0.2, 0)

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
    features_train = []
    features_test = []
    answers_train = []
    answers_test = []
    lab2_prepareDataset("Spam", ALL_SPAM_PRIMITIVE, 5, 0.2, 0)

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
    #Split the dataset
    trainDataset = []
    testDataset = []
    splitDataset(imgDataset, True, "")
    print(trainDataset)
    print("\nTEST\n")
    print(testDataset)

    #loadAllImages(r"data\csv\galaxy\galaxy_label_data_set_test.csv", r"data\images") #test
    #loadAllImages(r"data\csv\galaxy\galaxy_label_data_set.csv", r"data\images") #prod

    #B/W ratio
    feature1x = []
    feature1y = []
    #circularity
    feature2x = []
    feature2y = []
    #convexity
    feature3x = []
    feature3y = []
    #Bounding factor
    feature4x = []
    feature4y = []

    trainArray= []
    trainArrayLabels = []
    for ig in trainDataset:
        #print(ig.Answer, str(ig.BlackWhiteRatio), str(ig.C), str(ig.Convexity), str(ig.B),sep='--')
        trainArray.append([ig.BlackWhiteRatio, ig.C, ig.Convexity, ig.B])
        trainArrayLabels.append(ig.Answer)

        if "smooth" in ig.Answer:
            feature1x.append(ig.BlackWhiteRatio)    #BW ratio
            feature1y.append(ig.C)                  #Circularity
            feature3x.append(ig.Convexity)          #Convexity
            feature3y.append(ig.B)                  #Bounding rect fill factor
        else:
            feature2x.append(ig.BlackWhiteRatio)    #BW ratio
            feature2y.append(ig.C)                  #Circularity
            feature4x.append(ig.Convexity)          #Convexity
            feature4y.append(ig.B)                  #Bounding rect fill factor
        
    #traceGraph(feature1x, feature1y, "smooth", feature4x, feature4y, "spiral", "Black/White ratio", "Bounding ratio")
    #traceGraph(feature3x, feature3y, "smooth", feature2x, feature2y, "spiral", "Convexity", "Circularity")
    #traceGraph(feature1x, feature1y, "smooth", feature2x, feature2y, "spiral", "Black/White ratio", "Bounding ratio")
    #traceGraph(feature3x, feature3y, "smooth", feature4x, feature4y, "spiral", "Convexity", "Circularity")
    """
    print("END -----------------------------------------------------------")

__authors__ = [
    'Amhad Al-Tahter',
    'Jean-Philippe Decoste'
]
__copyright__ = 'École de technologie supérieure 2018'
