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

import numpy as np

from mdlp.discretization import MDLP
from commons.helpers.dataset.strategies.galaxy_dataset.feature_strategy import GalaxyDataSetFeatureStrategy


class SupervisedDiscretizationStrategy(object):
    """
        A class used for supervised data discretization.
    """

    def __init__(self):
        self.transformer = MDLP()

    def discretize(self, data_set, validation_size, nb_bins=None):
        """ Discretize continuous attribute using MDLP method.

        Args:
            data_set: The data set containing continuous data.
            validation_size: The validation size of the newly created discretized data set.

        Returns:
            discretized_dataset: A DataSet object containing discretized data.
        """

        # Create strategy object to further create the discretized data set.
        galaxy_dataset_feature_strategy = GalaxyDataSetFeatureStrategy()

        # Get data from training set.
        X_train = data_set.train.get_features
        y_train = data_set.train.get_labels

        # Supervised discretization of the training data set using MDLP.
        X_train_discretized = self.transformer.fit_transform(X=X_train, y=y_train)

        # Get data from validation set.
        X_valid = data_set.valid.get_features
        y_valid = data_set.valid.get_labels

        # Unsupervised discretization using MDLP.
        X_valid_discretized = self.transformer.transform(X=X_valid)

        # Merge both training and validation data.
        X = np.append(X_train_discretized, X_valid_discretized, axis=0)
        y = np.append(y_train, y_valid, axis=0)

        # Create a new data set.
        discretized_dataset = galaxy_dataset_feature_strategy.create_datasets(X, y, validation_size)

        return discretized_dataset
