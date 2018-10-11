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

import cv2 as cv
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import face, imread, imshow
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from sklearn import preprocessing, tree
from sklearn.datasets import load_iris


class Image:
    def __init__(self, path, label, answer):
        """
             The only construct is to build an object type image that has all the elements required to make a decision weather it is
             a smooth or spiral galaxy
             
             Args:
                 self : refers to the class
                 path : Where the image is stored
                 label : The image name 
                 answer : The final answer (smooth or spiral). This is the answer from the data set. 
                          It is used to verify after making a decision. 
                          
              Returns:
                  An Image object with the image manipulations
         """
        self.Path = path
        self.Label = label
        self.Answer = answer
        self.Image = cv.imread(path)
        self.Pixels = np.array(self.Image)
        self.Width = self.Pixels.shape[0]
        self.Height = self.Pixels.shape[1]
        #we will always use the cropped image
        #default crop is 250
        #self.crop(self.Width) #no crop
        self.crop(200)

        #useful image manipulations
        self.manipulations()

        #default computations
        self.ComputeCircularity()
        self.computeBlackWhite()
        self.ComputeConvexity()
        self.ComputeBoundingRectangleToFillFactor()

        #plt.imshow(self.Image),plt.title('ORIGINAL')
        #plt.show()
        #cv.imshow('gray_image', self.remove_starlight()) 
        #plt.imshow(self.remove_starlight()),plt.title('GRAYSCALE')
        #plt.show()
        
    def manipulations(self):
        """
        This method is used to apply all the images manipulation on the image. Such as grayscale.
        Cropped image is the default image used 
        
        Args:
            self: refers to the class
        """
        #remove noise by blurring with a Gaussian filter
        self.CroppedPixels = cv.GaussianBlur(self.CroppedPixels,(3,3), 0)

        #convert to grayscale
        self.GrayScale = cv.cvtColor(self.CroppedPixels, cv.COLOR_BGR2GRAY)
        """remove background noise
        #the result is worst with laplacian
        #laplacian = cv.Laplacian(self._GrayScale, cv.CV_16S, 3)
        #laplacian = cv.convertScaleAbs(laplacian)
        #self._GrayScale = self._GrayScale - laplacian 
        """
        self.Threshold = threshold_otsu(self.GrayScale)
        self.Binary = self.GrayScale > self.Threshold
    
    def crop(self, dimentions):
        """
        This method is used to apply a crop to the image. Since the image is squared only on parameter is required 
        
        Args:
            dimentions: refers the final demention of the image. Such as the final image would have
                        dimentions*dimentions. Ex: dimentions=250 the image will be 250x250
        """
        # dimention is the width and Height to crop to. Since it is a square.
        upper_width = int(self.Width/2 + dimentions/2)
        lower_width = int(self.Width/2 - dimentions/2)
        upper_height = int(self.Height/2 + dimentions/2)
        lower_height = int(self.Height/2 - dimentions/2)
        #new array of the image
        self.CroppedPixels = self.Pixels[lower_width:upper_width, lower_height:upper_height]

    #All manipulations
    def remove_starlight(self):
        """ Removes the star light in images.

        Calclates the median in color and gray scale image to clean the image's background.

        Args:
             image_color: an OpenCV standard color image format.
             image_gray: an OpenCV standard gray scale image format.

        Returns:
            An image cleaned from star light.
        """
        t = np.max(np.median(self.Image[np.nonzero(cv.cvtColor(self.Pixels, cv.COLOR_BGR2GRAY))], axis=0))
        self.Image[self.Image < t] = t

        #return self.rescale(self.Image).astype("uint8")
        return self.rescale(self.Image)
        #return self.Image
    
    def rescale(self, image, min=20, max=255):
        """ Rescale the colors of an image.

        Utility method to rescale colors from an image. 

        Args: 
            image: an OpenCV standard image format.
            min: The minimum color value [0, 255] range.
            max: The maximum color value [0, 255] range.
        
        Returns:
            The image with rescaled colors.
        """
        image = image.astype('float')
        image -= image.min()
        image /= image.max()
        image = image * (max - min) + min

        return image
    
    def computeBlackWhite(self):
        """
        This method is used to compute the black and white ratio.
        The formula is blacks / whites
        
        Args:
            self: refers to the class
        """
        self.Black = 0
        self.White = 0
        self.BlackWhiteRatio = 0
        #compute the # of black and the # of whites
        for row in self.Binary:
            for pixel in row:
                if(pixel):
                    self.White += 1
                else:
                    self.Black += 1
        #compute the B/W ratio
        if self.Black > 0 and self.White > 0 :
            self.BlackWhiteRatio = self.Black / self.White
        else:
            self.BlackWhiteRatio = self.White / self.Black
        
    def ComputeCircularity(self):
        """
        This method is used to compute the circularity of the galaxy.
        The formula used is : 𝐶 = 4 ∗ 𝜋 ∗ 𝐴/𝑃2
        
        Args:
            self: refers to the class
         """
        #example from openCV documentation
        
        #thresh = cv.adaptiveThreshold(self.GrayScale, 100, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 501, 0)
        ret, thresh = cv.threshold(self.GrayScale, self.Threshold, 127, 0)
        img, contours, hierarchy = cv.findContours(thresh, 1 ,2)
        self.cnt = contours[0]
        #cv.imshow('img', img)
        
        self.area = cv.contourArea(self.cnt)
        self.perimeter = cv.arcLength(self.cnt,True)
        #circularity
        self.C = 0
        if self.area > 0 and self.perimeter > 0 :
            self.FormFactor = self.area/ math.pow(self.perimeter,2)
            self.C = 4 * math.pi * self.FormFactor 
    
    def ComputeConvexity(self):
        """
        This method is used to compute the convexity of the galaxy.
        The formula used is : 𝐶 = P / (2W + 2H)
        where P is the perimeter
        W is the width of the bounding rectangle
        H is the height of the bounding rectangle
        
        Args:
            self: refers to the class
         """
        x,y,w,h = cv.boundingRect(self.cnt)
        self.Convexity = self.perimeter / (2*w+2*h)

    def ComputeBoundingRectangleToFillFactor(self):
        """
        This method is used to compute the bounding rectangle to fill factor.
        The formula used is : B = A / (W*H)
        Where is A is the area of the shape
        W*H is the area of the bounding rectangle
        Args:
            self: refers to the class
         """
        x, y, w, h = cv.boundingRect(self.cnt)
        self.B = self.area / (w*h)
#############################################################################################
def loadAllImages(dataPath, imageFolderPath):
    """
    This method is used to al the images from the data set.
    
    Args:
        dataPath: the dataset file path
        imageFolderPath: the image dataset folder path
    """
    dataFile = open(dataPath, 'r') # option r veut dire read
    
    #index is used to load a define number of images.
    index = 1
    #Skip header
    next(dataFile)
    
    for line in dataFile:
        texts = line.split(",")
        imageName = texts[0]
        shape = str(texts[1])
        imagePath = imageFolderPath +"\\"+str(imageName)+'.jpg'
        
        if index % 3 != 2:
            trainDataSet.append(Image(imagePath,imageName,shape))
            index += 1
        else:
            testDataSet.append(Image(imagePath,imageName,shape))
            index += 1

    dataFile.close()
    
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

def buildTree(trainArray, trainArraylabels):
    """
    This method is used to build the decision tree and how the graph associated with it.
    fit(x,y) takes two arrays:
    X, sparse or dense, of size [n_samples, n_features] holding the training samples, and an array 
            ex: [[0, 0], [1, 1]]
    Y of integer values, size [n_samples], holding the class labels 
            ex : [0, 1]
    
    Since out Y array is not numerical we are using preprocessing from sklearn to transform them
    Args:
        trainArray: the array containing all the features. It will be used as the X
        trainArraylabels: the array containing all the labels. It will be used as the Y
    """
    #Encode label as 0 and 1 (respectively)
    le = preprocessing.LabelEncoder()
    le.fit(trainArraylabels)
    y = le.transform(trainArraylabels)
    #print(trainArraylabels)
    #print(y)

    #Build the tree
    clf = tree.DecisionTreeClassifier(max_depth = TREE_DEPTH)
    clf = clf.fit(trainArray, y)

    #Create graph. If executed localy, need to install Graphviz and set its \bin folder to System PATH
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("Galaxy")

    #Return the tree for testing later
    return clf


TREE_DEPTH = 10

# Main
if __name__ == '__main__':
    #sepration 70%-30% of the data set
    trainDataSet = []
    testDataSet = []

    #loadAllImages(r"data\csv\galaxy\galaxy_label_data_set_test.csv", r"data\images") #test
    loadAllImages(r"data\csv\galaxy\galaxy_label_data_set.csv", r"data\images") #prod

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
    trainArraylabels = []
    for ig in trainDataSet:
        #print(ig.Answer, str(ig.BlackWhiteRatio), str(ig.C), str(ig.Convexity), str(ig.B), sep=' -- ')
        trainArray.append([ig.BlackWhiteRatio, ig.C, ig.Convexity, ig.B])
        trainArraylabels.append(ig.Answer)

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
    traceGraph(feature1x, feature1y, "smooth", feature2x, feature2y, "spiral", "Black/White ratio", "Bounding ratio")
    traceGraph(feature3x, feature3y, "smooth", feature4x, feature4y, "spiral", "Convexity", "Circularity")

    theTree = buildTree(trainArray, trainArraylabels)

    #Testing part
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
    print("END -----------------------------------------------------------")

__authors__ = [
    'Amhad Al-Tahter',
    'Jean-Philippe Decoste'
]
__copyright__ = 'École de technologie supérieure 2018'
