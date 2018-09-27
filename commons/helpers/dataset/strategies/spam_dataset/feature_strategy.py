#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cours :
    GTI770 — Systèmes intelligents et apprentissage machine

Projet :
    Laboratoire 1 — Extraction de primitives

Étudiants :
    Noms — Code permanent

Groupe :
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


class SpamDataSetFeatureStrategy:
    """
        A class for handling data set files.
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

    def _create_datasets(self, spam_features, labels, validation_size):
        # Creates inner DataSets class.
        class DataSets(object):
            pass

        # Create an instance of a DataSets object.
        data_sets = DataSets()

        # Check if the parameter is of type numpy.float64.
        self._is_type(validation_size)
        self._is_positive(validation_size)

        # Calculates the training set and validation set size.
        train_size = int(np.round((1 - validation_size) * spam_features.shape[0]))
        validation_size = int(np.round(validation_size * spam_features.shape[0]))

        # Assign the images to the training and validation data sets.
        train_spam_features = spam_features[:train_size]
        train_labels = labels[:train_size]
        validation_spam_features = spam_features[-validation_size:]
        validation_labels = labels[-validation_size:]

        # Create the data sets.
        data_sets.train = DataSet().withFeatures(train_spam_features).withLabels(train_labels)
        data_sets.valid = DataSet().withFeatures(validation_spam_features).withLabels(validation_labels)

        return data_sets

    def _load_feature_vector(self, csv_file, one_hot):
        """ Read the feature vector of a spam sample.

        Read the label associated to a feature vector in the reference .arff file.

        Args:
            arff_file (str): The file path to the .arff file containing the ground truth.

        Returns:
            A tuple with the extracted feature vectors and their associated class labels.

        """

        class DataSets(object):
            pass

        data_sets = DataSets()

        # Declare two lists for holding spam features and the associated class label.
        spam_vectors = list()
        labels = list()
        encoded_labels = list()

        try:
            # Open the ground truth file.
            with open(csv_file, mode="r") as ground_truth_csv:
                reader = csv.reader(ground_truth_csv, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)

                # For each row, store the spam feature vector and its associated class.
                for row in reader:
                    spam_vectors.append(row[0:57])
                    labels.append(row[57])

        except FileNotFoundError:
            raise FileNotFoundException("CSV file not found. Please enter in parameter a valid CSV file.")

        # Transforms the feature list into a numpy array.
        spam_vectors = np.array(spam_vectors)

        # Reshape vertically the label vector and transform it in a numpy array.
        labels = np.array(labels).reshape(-1, 1)

        # Declare a label encoder from scikit-learn.
        label_encoder = LabelEncoder()

        # Fit label encoder and return the classes encoded into integer in range [0,2]
        encoded_labels = label_encoder.fit_transform(labels)

        if one_hot == True:
            # Declare a one-hot encoder from scikit-learn.
            one_hot_encoder = OneHotEncoder(sparse=False)

            encoded_labels = encoded_labels.reshape(-1, 1)

            # Fit label encoder and return the classes encoded into integer in range [0,2]
            encoded_labels = one_hot_encoder.fit_transform(encoded_labels)

        # Shuffle the data.
        features, encoded_labels = shuffle(spam_vectors, encoded_labels)

        return features, encoded_labels

    def load_dataset(self, csv_file, one_hot, validation_size):
        """ Load a data set.

             Args:
                 csv_file: a CSV file containing ground truth and file names.
                 feature_vector: a boolean. It True, will load the data set from a feature vector.
                                 If False, will load the data set required to extract galaxy image features.

             Returns:
                 A tuple containing the feature vectors and labels associated to these vectors.
             """
        try:
            feature_vectors, labels = self._load_feature_vector(csv_file, one_hot)
            return self._create_datasets(feature_vectors, labels, validation_size)

        except Exception as e:
            raise UnableToLoadDatasetException("Unable to load spam data set with cause: " + str(e))
