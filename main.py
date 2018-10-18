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

import math
import random

import graphviz
import matplotlib.pyplot as plt
from image import Image
from sklearn import preprocessing, tree
#from sklearn.datasets import load_iris


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
    for line in dataFile:
        #Prepare data
        texts = line.split(",")
        imageName = texts[0]
        shape = str(texts[1])
        imagePath = imageFolderPath + "\\" + str(imageName) + '.jpg'
        #Push data
        tempDataset.append(Image(imagePath, imageName, shape))

    dataFile.close()

    return tempDataset

def splitDataset(og_dataset, shuffle, option):
    """
    This method is used to al the images from the data set.

    Args:
        og_dataset: the dataset
        shuffle: do shuffle before split
        option: how to split the file in 2
    """
    limit = len(og_dataset)

    if shuffle:
        random.shuffle(og_dataset)

    if option == "test":
        limit = TEST_NBSAMPLES

    for i in range(0, limit):
        if i % 3 != 2:
            trainDataSet.append(og_dataset[i])
        else:
            testDataSet.append(og_dataset[i])

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

def buildTree(data, labels):
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
    #Encode label as 0 and 1 (respectively)
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(labels)
    y = labelEncoder.transform(labels)
    #print(labels)
    #print(y)

    #Build the tree
    clf = tree.DecisionTreeClassifier(max_depth=TREE_DEPTH)
    clf = clf.fit(data, y)

    #Create graph. If executed localy, need to install Graphviz 
    # and set its \bin folder to System PATH
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("Galaxy")

    #Return the tree for testing later
    return clf

TEST_NBSAMPLES = 50
TREE_DEPTH = 10
SMOOTH_GALAXY = 0
SPIRAL_GALAXY = 1

# Main
if __name__ == '__main__':
    #Filling imgDataset with all image referenced in CSV
    imgDataset = loadAllImages(r"data\csv\galaxy\galaxy_label_data_set_test.csv", r"data\images") #test
    #imgDataset = loadAllImages(r"data\csv\galaxy\galaxy_label_data_set.csv", r"data\images") #prod

    #Split the dataset
    trainDataSet = []
    testDataSet = []
    splitDataset(imgDataset, True, "")
    print(trainDataSet)
    print("\nTEST\n")
    print(testDataSet)

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
    for ig in trainDataSet:
        #print(ig.Answer, str(ig.BlackWhiteRatio), str(ig.C), str(ig.Convexity), str(ig.B), sep=' -- ')
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

    #theTree = buildTree(trainArray, trainArrayLabels)

    #Testing part
    """
    success_percent = 0
    testArray = []
    testArraylabels = []
    for ig in testDataSet:
        testArray.append([ig.BlackWhiteRatio, ig.C, ig.Convexity, ig.B])
        testArraylabels.append(ig.Answer)

    
    predictions = theTree.predict(testArray)
    print(predictions)
    print(testDataSet)

    index = 0
    for prediction in predictions:
        print(prediction, testArraylabels[index], sep=' -- ')
        if prediction == 0 and "smooth" in testArraylabels[index]:
            success_percent += 1
        elif prediction == 1 and "spiral" in testArraylabels[index]:
            success_percent += 1
        index += 1
    
    print(str(success_percent) + "/" + str(len(testDataSet)))
    """
    print("END -----------------------------------------------------------")

__authors__ = [
    'Amhad Al-Tahter',
    'Jean-Philippe Decoste'
]
__copyright__ = 'École de technologie supérieure 2018'
