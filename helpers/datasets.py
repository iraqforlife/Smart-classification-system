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
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import decomposition
from helpers import utilities as utils
from helpers import image as imageObj
from helpers import imageV2 as imageVectors
from helpers import imageV2 as imageVectors_SVM
from helpers import spam
from helpers import music


#LAB01
IMAGE_CSV_NAME = r"data\csv\galaxy\galaxy_label_data_set.csv"
#IMAGETEST_CSV_NAME = r"data\csv\galaxy\galaxy_label_data_set_test.csv"

#Galaxy features
EXTRATED_GALAXY_PRIMITIVE = r"data\csv\eq07_pExtraction.csv"
MERGED_GALAXY_PRIMITIVE = r"data\csv\eq07_pMerged.csv"
ALL_GALAXY_PRIMITIVE = r"data\csv\galaxy\galaxy_feature_vectors.csv"

#Spam features
ALL_SPAM_PRIMITIVE = r"data\csv\spam\spam.csv"


####################
#       LAB01      #
####################
def extractFeatures(skipHeader, scanPrimitives, doMerge):
    destFile_rowCount = 0
    sourceFile_rowCount = 0
    allGalaxy_length = len(list(csv.reader(open(ALL_GALAXY_PRIMITIVE))))

    #If source images csv row count > extracted features csv row count, 
    # we process each images in the file
    if os.path.isfile(EXTRATED_GALAXY_PRIMITIVE):
        destFile_rowCount = len(list(csv.reader(open(EXTRATED_GALAXY_PRIMITIVE))))
        sourceFile_rowCount = len(list(csv.reader(open(IMAGE_CSV_NAME)))) - 1

    if scanPrimitives and sourceFile_rowCount > destFile_rowCount:
        #imgDataset = loadAllImages(skipHeader, IMAGETEST_CSV_NAME, r"data\images") #test
        imgDataset = loadAllImages(skipHeader, IMAGE_CSV_NAME, r"data\images") #prod

        with open(EXTRATED_GALAXY_PRIMITIVE, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(imgDataset)

    #Merge feature we extracted
    if doMerge:
        print("MERGE")
        valueFirstFile = []
        mergedFiles = []
        progress = 0

        print("Reading file '" + EXTRATED_GALAXY_PRIMITIVE + "'")
        utils.printProgressBar(0, destFile_rowCount, prefix='Progress:', suffix='Complete', length=50)
        with open(EXTRATED_GALAXY_PRIMITIVE, "r") as newFeature:
            reader = csv.reader(newFeature, delimiter=',', quotechar='|')
            for row in reader:
                progress += 1
                utils.printProgressBar(progress+1, destFile_rowCount, prefix='Progress', suffix='Complete', length=50)

                values = [float(i) for i in row]
                valueFirstFile.append(values)
        print("\n-> Done\n")

        progress = 0
        print("Reading file '" + ALL_GALAXY_PRIMITIVE + "' and merging it with the first one")
        utils.printProgressBar(0, allGalaxy_length, prefix='Progress:', suffix='Complete', length=50)
        with open(ALL_GALAXY_PRIMITIVE, "r") as allFeature:
            reader = csv.reader(allFeature, delimiter=',', quotechar='|')
            for row in reader:
                progress += 1
                utils.printProgressBar(progress+1, allGalaxy_length, prefix='S', suffix='DONE', length=50)

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

def loadAllImages(skipHeader, dataPath, imageFolderPath):
    """
    This method is used to al the images from the data set.

    Args:
        dataPath: the dataset file path
        imageFolderPath: the image dataset folder path
    """
    #Opened CSV file (r = read)
    dataFile = open(dataPath, 'r')
    #Skip header
    if skipHeader:
        next(dataFile)

    tempDataset = []
    fileLength = len(list(csv.reader(open(dataPath))))
    # Initial call to print 0% progress
    progress = 0
    print("Extracting our own features:")
    utils.printProgressBar(0, fileLength, prefix='Progress:', suffix='Complete', length=50)
    for line in dataFile:
        #Prepare data
        texts = line.split(",")
        imgName = texts[0]
        shape = str(texts[1])
        imgPath = imageFolderPath + "\\" + str(imgName) + '.jpg'

        #Print progress
        progress += 1
        #sleep(0.1)
        utils.printProgressBar(progress + 1, fileLength, prefix='Progress:', suffix='Complete', length=50)

        #The image
        img = imageObj(imgPath, imgName, shape)

        #Push data
        tempDataset.append([img.Label, img.BlackWhiteRatio, img.C, img.Convexity, img.B])

    dataFile.close()

    return tempDataset



####################
#       LAB02      #
####################
def prepareDatasetLab2(datasetName, dataset, n_splits, test_size, random_state):
    print("PREPARING DATASETS")
    allData_length = len(list(csv.reader(open(dataset))))
    progress = 0
    datas = []

    print("Reading " + datasetName + " features:")
    utils.printProgressBar(0, allData_length, prefix='Progress:', suffix='Complete', length=50)
    with open(dataset, 'r') as theFile:
        primitives = csv.reader(theFile, delimiter=',', quotechar='|')

        for row in primitives:
            progress += 1
            utils.printProgressBar(progress+1, allData_length, prefix='Progress', suffix='Complete', length=50)

            values = [float(i) for i in row]
            if datasetName == "Galaxy":
                datas.append(imageVectors.Image(values))
            elif datasetName == "Spam":
                datas.append(spam.Spam(values))
    print("\n-> Done\n")

    #3. Split dataset using model_selection
    print("Splitting Dataset according to these params:")
    print(tabulate([['Property', 'Value'], ['n_splits', n_splits], ['test_size', test_size], ['random_state', random_state]], headers="firstrow"))

    features = []
    features_SVM = []
    answers = []
    for data in np.array(datas):
        features_SVM.append([data.features[4], data.features[5], data.features[6], data.features[19], data.features[23], data.features[24]])
        features.append(data.features)
        answers.append(data.answer)
    
    #4. Scale data
    dataScaler = MinMaxScaler()
    dataScaler.fit(features)
    features_scaled = dataScaler.transform(features)

    dataScaler.fit(features_SVM)
    features_SVM_scaled = dataScaler.transform(features_SVM)

    dataset_splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    print("-> Done\n\n")

    return features_scaled, features_SVM_scaled, answers, dataset_splitter


####################
#       LAB04      #
####################
def musicFeatures(row):
    '''
        Transform the CSV file data into useful data
        
        params:
            row: the csv row with all the data 
        
        return:
            feature list 
            label
    '''
    answer = row[-1]
    musicId = row[1]
    #remove the answer
    del row[-1]
    #remove the identifier (ex:27)
    del row[0]
    #remove The id (ex:TRAAAAK128F9318786)
    del row[0]

    features = [float(i) for i in row]
    
    return features,answer,musicId

def prepareTrainDataset(datasetName, dataset):
    print("PREPARING DATASETS")
    allData_length = len(list(csv.reader(open(dataset))))
    progress = 0
    
    features = []
    answers = []

    print("Reading " + datasetName + " features:")
    utils.printProgressBar(0, allData_length, prefix='Progress:', suffix='Complete', length=50)
    with open(dataset, 'r') as theFile:
        primitives = csv.reader(theFile, delimiter=',', quotechar='|')

        for row in primitives:
            progress += 1
            utils.printProgressBar(progress+1, allData_length, prefix='Progress', suffix='Complete', length=50)

            feature, answer, mId = musicFeatures(row)            
            features.append(feature)
            answers.append(answer)
    
    #4. Scale data
    dataScaler = MinMaxScaler()
    dataScaler.fit(features)
    features_scaled = dataScaler.transform(features)
    #5 PCA
    transformed =  decomposition.PCA(n_components=0.95).fit_transform(features_scaled)
    #6 Transform labels
    le = LabelEncoder()
    le.fit(answers)
    answers = le.transform(answers)
    print("\n-> Done\n")
    return transformed , answers, le
    
def splitDataSet(features,answers, n_splits, test_size, random_state):
    
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
        
    for train_index, test_index in splited:
        features_train, features_test = features[train_index], features[test_index]
        answers_train, answers_test = answers[train_index], answers[test_index]
    
    print("-> Done\n\n")

    return features_train,answers_train,features_test,answers_test

def prepareTestDataset(datasetName, dataset):
    print("PREPARING DATASETS")
    allData_length = len(list(csv.reader(open(dataset))))
    progress = 0
    
    features = []
    ids=[]

    print("Reading " + datasetName + " features:")
    utils.printProgressBar(0, allData_length, prefix='Progress:', suffix='Complete', length=50)
    with open(dataset, 'r') as theFile:
        primitives = csv.reader(theFile, delimiter=',', quotechar='|')

        for row in primitives:
            progress += 1
            utils.printProgressBar(progress+1, allData_length, prefix='Progress', suffix='Complete', length=50)

            feature, answer, mId = musicFeatures(row)            
            features.append(feature)
            ids.append(mId)
    
    #4. Scale data
    dataScaler = MinMaxScaler()
    dataScaler.fit(features)
    features_scaled = dataScaler.transform(features)
    #5 PCA
    transformed =  decomposition.PCA(n_components=0.95).fit_transform(features_scaled)
    print("\n-> Done\n")
    return transformed , ids

def outPut(ids,answers):    
    assert(len(ids)==len(answers))
    file = open("output.csv", "w")
    
    for index in range(0,len(ids)):
        file.write("%s;%s" % (ids[index],answers[index]))
        file.write("\n")
    print("file creation is done2.")   