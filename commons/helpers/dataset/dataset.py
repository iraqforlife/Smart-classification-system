#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # X - Lab's name

Students :
    Names — Permanent Code

Group :
    GTI770-H18-0X
"""

import cv2
import numpy as np
import os

class DataSet(object):
    """
        An object for storing data set elements.
    """

    def __init__(self):
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def get_images(self):
        return self._images

    @property
    def get_features(self):
        return self._features

    @property
    def get_labels(self):
        return self._labels

    @property
    def get_num_examples(self):
        return self._num_examples

    @property
    def get_epochs_done(self):
        return self._epochs_done

    def withLabels(self, labels):
        self._labels = labels
        self._num_examples = labels.shape[0]
        return self

    def withImg_names(self, img_names):
        self._img_names = img_names
        self._num_examples = img_names.shape[0]
        return self

    def withFeatures(self, features):
        self._features = features
        self._num_examples = features.shape[0]
        return self

    def withImages(self, images):
        self._images = images
        self._num_examples = images.shape[0]
        return self

    def next_feature_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.

        Args:
            batch_size: the number of element in the batch.

        Returns:
            A tuple containing a list of feature vectors and the associated labels.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_done += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]

    def next_image_batch(self, batch_size):
        """
        Return the next `batch_size` examples from this data set.

        Args:
            batch_size: the number of element in the batch.

        Returns:
            A tuple containing a list of img_names (i.e. 1000742) and the associated labels.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_done += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            self._img_names = self._img_names[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._img_names[start:end], self._labels[start:end]

    def load_images(self, batch):
        """ Load a training image data set.

        Args:
            batch: A list of IDs used in the image file names to load.

        Return:
             The loaded images.
        """

        # Declare a list for storing OpenCV images.
        images = list()

        for sample in batch:
            # Create path to image's file.

            path = (os.environ["VIRTUAL_ENV"] + "/data/images/" + str(sample[0]) + ".jpg")

            # Use OpenCV to read the image.
            image = cv2.imread(path)

            # Transform image as a 32-bit numpy array.
            image = image.astype(np.float32)

            # Normalize the image.
            image = np.multiply(image, 1.0 / 255.0)

            # Append the image's name.
            images.append(image)

        # Put all the loaded images into a numpy array.
        images = np.array(images)

        self._images = images

        return self._images
