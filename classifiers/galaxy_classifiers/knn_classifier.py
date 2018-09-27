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

from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(object):
    """ An object containing a decision tree classifier. """

    def __init__(self, nb_neighbors, weights):
        self.model = KNeighborsClassifier()
        self.model.n_neighbors = nb_neighbors
        self.model.weights = weights

    def standardize(self, X):
        """ Standardize the data.

        Args:
            X: The input vector [n_sample, n_feature].

        Returns:
            X: The input vector with standardized values.
        """

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = ((X - mean) / std)

        return X