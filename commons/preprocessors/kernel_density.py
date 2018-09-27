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

from sklearn.neighbors import KernelDensity


class GaussianKernelDensity(object):

    def __init__(self):
        self.kde = KernelDensity(bandwidth=1.0, algorithm="auto", kernel="gaussian")

    def train(self, X):
        """ Fit the Kernel Density model on the data

        Args:
            X: A 2-D array of data.
        """

        self.kde.fit(X=X)

    def score_samples(self, X):
        """ Evaluate the density model on the data.

        Args:
            X: A 2-D array of data.

        Returns:
            The array of log(density) evaluations.
        """

        return self.kde.score_samples(X=X)
