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
import csv

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle

from commons.exceptions.fileNotFoundException import FileNotFoundException
from commons.exceptions.unableToLoadDatasetException import UnableToLoadDatasetException
from commons.exceptions.validationSizeException import ValidationSizeException
from commons.helpers.dataset.dataset import DataSet


class GalaxyDataSetImageStrategy:
    """
        A class for handling data set of images files.
    """

    def _is_positive(self, number):
        """ Verify the type of a variable.

        Make a comparison of the type of a variable to insure it is a float.

        Args:
            number: a floating point number.

        Returns:
            A boolean; true if number passed in argument is of type numpy.float32, false if type is not matched.
        """

        try:
            if number < 0.0:
                raise ValidationSizeException(
                    "Validation size must be a positive floating point number or equals to 0.0.")
            if number >= 0.0:
                return number

        except AttributeError:
            raise ValidationSizeException("Validation size is not a valid floating point number.")

    def _is_type(self, number, type=np.float32):
        """ Verify the type of a variable.

        Make a comparison of the type of a variable to insure it is a float.

        Args:
            number: a floating point number.

        Returns:
            A boolean; true if number passed in argument is of type numpy.float32, false if type is not matched.
        """

        try:
            return number.dtype.num == np.dtype(type).num

        except AttributeError:
            raise ValidationSizeException("Validation size is not a valid floating point number.")

    def _create_datasets(self, img_names, labels, validation_size):
        """ Create a data set.

          Args:
              img_names: The file name of the images.
              labels: The associated label (class) of images.
              validation_size: The desired amount of samples needed to build the validation set.

          Returns:
              data_sets: A DataSet object containing data from previously parsed CSV files.
        """

        # Creates inner DataSets class.
        class DataSets(object):
            pass

        # Create an instance of a DataSets object.
        data_sets = DataSets()

        # Check if the parameter is of type numpy.float64.
        self._is_type(validation_size)
        self._is_positive(validation_size)

        # Calculates the training set and validation set size.
        train_size = int(np.round((1 - validation_size) * img_names.shape[0]))
        validation_size = int(np.round(validation_size * img_names.shape[0]))

        # Assign the images to the training and validation data sets.
        train_img_names = img_names[:train_size]
        train_labels = labels[:train_size]
        validation_img_names = img_names[-validation_size:]
        validation_labels = labels[-validation_size:]

        # Create the data sets.
        data_sets.train = DataSet().withImg_names(train_img_names).withLabels(train_labels)
        data_sets.valid = DataSet().withImg_names(validation_img_names).withLabels(validation_labels)

        return data_sets

    def _read_labels(self, csv_file, one_hot):
        """ Read the labels associated to a galaxy ID.

        Read the label associated to a galaxy ID in the reference CSV file.

        Args:
            csv_file (str): The file path to the CSV file containing the ground truth.

        Returns:
            The extracted and shuffled galaxy IDs and their associated encoded labels.

        """

        # Declare two lists for holding galaxy ID reference and the associated class label.
        galaxy_ids = list()
        labels = list()
        encoded_labels = list()

        try:
            # Open the ground truth file.
            with open(csv_file, mode="r") as ground_truth_csv:
                reader = csv.reader(ground_truth_csv, delimiter=",")

                # For each row, store the galaxy ID and its associated class.
                for row in reader:
                    if row == ["id", "class"]:
                        continue
                    galaxy_ids.append(row[0])
                    labels.append(row[1])

        except FileNotFoundError:
            raise FileNotFoundException("CSV file not found. Please enter in parameter a valid CSV file.")

        # Transforms the lists into a vertical numpy array.
        galaxy_ids = np.array(galaxy_ids).reshape(-1, 1)
        labels = np.array(labels).reshape(-1, 1)

        # Declare a label encoder from scikit-learn.
        label_encoder = LabelEncoder()

        # Fit label encoder and return the classes encoded into integer in range [0,2]
        encoded_labels = label_encoder.fit_transform(labels).reshape(-1, 1)

        if one_hot == True:
            # Declare a one-hot encoder from scikit-learn.
            one_hot_encoder = OneHotEncoder(sparse=False)

            encoded_labels = encoded_labels.reshape(-1, 1)

            # Fit label encoder and return the classes encoded into integer in range [0,2]
            encoded_labels = one_hot_encoder.fit_transform(encoded_labels)

        # Shuffle the data.
        galaxy_ids, encoded_labels = shuffle(galaxy_ids, encoded_labels)

        return galaxy_ids, encoded_labels

    def load_dataset(self, csv_file, one_hot, validation_size):
        """ Load a data set.

        Args:
            csv_file: a CSV file containing ground truth and file names.
            one_hot: a boolean. It True, will load the data set labels as a one-hot vector e.g. [0, 1, 0].
                            If False, will load the data set labels as integers.
            validation_size: the specified user's validation data set size.

        Returns:
            A DataSet object containing a training and validation part of image vectors.
        """
        try:
            img_names, labels = self._read_labels(csv_file, one_hot)
            return self._create_datasets(img_names, labels, validation_size)

        except Exception as e:
            raise UnableToLoadDatasetException("Unable to load galaxies data set with cause: " + str(e))
