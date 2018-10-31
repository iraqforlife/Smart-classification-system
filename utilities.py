import csv
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from imageV2 import Image as imageFeat
from spam import Spam
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
PRIMITIVE_SCANNING = True
DOMERGE = False


####################
#       LAB01      #
####################
def lab1_extractFeatures(self):
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
        imgDataset = self.loadAllImages(IMAGE_CSV_NAME, r"data\images") #prod

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
        self.printProgressBar(0, destFile_rowCount, prefix='Progress:', suffix='Complete', length=50)
        with open(EXTRATED_GALAXY_PRIMITIVE, "r") as newFeature:
            reader = csv.reader(newFeature, delimiter=',', quotechar='|')
            for row in reader:
                progress += 1
                self.printProgressBar(progress+1, destFile_rowCount, prefix='Progress', suffix='Complete', length=50)

                values = [float(i) for i in row]
                valueFirstFile.append(values)
        print("\n-> Done\n")

        progress = 0
        print("Reading file '" + ALL_GALAXY_PRIMITIVE + "' and merging it with the first one")
        self.printProgressBar(0, allGalaxy_length, prefix='Progress:', suffix='Complete', length=50)
        with open(ALL_GALAXY_PRIMITIVE, "r") as allFeature:
            reader = csv.reader(allFeature, delimiter=',', quotechar='|')
            for row in reader:
                progress += 1
                self.printProgressBar(progress+1, allGalaxy_length, prefix='Progress', suffix='Complete', length=50)

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

def loadAllImages(self, dataPath, imageFolderPath):
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
    self.printProgressBar(0, fileLength, prefix='Progress:', suffix='Complete', length=50)
    for line in dataFile:
        #Prepare data
        texts = line.split(",")
        imgName = texts[0]
        shape = str(texts[1])
        imgPath = imageFolderPath + "\\" + str(imgName) + '.jpg'

        #Print progress
        progress += 1
        #sleep(0.1)
        self.printProgressBar(progress + 1, fileLength, prefix='Progress:', suffix='Complete', length=50)

        #The image
        img = imageObj(imgPath, imgName, shape)

        #Push data
        tempDataset.append([img.Label, img.BlackWhiteRatio, img.C, img.Convexity, img.B])

    dataFile.close()

    return tempDataset

def splitDataset(self, og_dataset, shuffle, option):
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

def traceGraph(self, feature1x, feature1y, feature1Name, feature2x, feature2y, feature2Name, xlabel, ylabel):
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
###################

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

def printGraph(self):
    #fig = plt.figure()
    #ax1 = fig.add_subplot()
    plt.grid(True)
    #plt.scatter(feature1x, feature1y, s=10, c='b', marker="s", label=feature1Name)
    #plt.scatter(feature2x, feature2y, s=10, c='r', marker="o", label=feature2Name)
    #plt.ylabel(xlabel)
    #plt.xlabel(ylabel)
    #plt.legend(loc='upper left')
    #plt.show()