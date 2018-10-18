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