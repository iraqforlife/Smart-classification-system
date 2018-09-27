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

from __future__ import division, print_function, absolute_import

from sklearn.svm import LinearSVC


class LinearSVMClassifier(object):
    """ An object containing a linear support vector machine classifier. """

    def __init__(self, C, class_weight):
        self.model = LinearSVC()
        self.model.C = C
        self.model.class_weight = class_weight

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