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

from commons.helpers.dataset.strategies.galaxy_dataset.feature_strategy import GalaxyDataSetFeatureStrategy

class UnsupervisedDiscretizationStrategy(object):

    def __init__(self):
        pass

    def find_range(self, X):
        """ Find the range in values of features..

        Args:
            X: A data set of continuous features.

        Returns:
            A tuple representing the minimum and maximum value of the feature.
        """

        min = np.amin(X, axis=0)
        max = np.amax(X, axis=0)

        return min, max

    def discretize(self, data_set, validation_size, nb_bins=10):
        """ Discretize continuous values into bins using unsupervised algorithm.

        Args:
            data_set: The data set containing continuous data.
            validation_size: The validation size of the newly created discretized data set.

        Returns:
            discretized_dataset: A DataSet object containing discretized data.
        """

        galaxy_dataset_feature_strategy = GalaxyDataSetFeatureStrategy()

        X = np.append(data_set.train.get_features, data_set.valid.get_features, axis=0)
        y = np.append(data_set.train.get_labels, data_set.valid.get_labels, axis=0)

        digitized = np.zeros(shape=X.shape)

        for i in range(0, X[0].size):

            min, max = self.find_range(X[:, i])

            bins = np.linspace(min, max, nb_bins)

            digitized[:, i] = (np.digitize(X[:, i], bins))

        discretized_dataset = galaxy_dataset_feature_strategy.create_datasets(digitized, y, validation_size)

        return discretized_dataset