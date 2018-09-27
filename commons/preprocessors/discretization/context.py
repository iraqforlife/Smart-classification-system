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


class DiscretizerContext:
    """
        Define the interface of interest to clients. Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def discretize(self, data_set, validation_size, nb_bins=10):
        """ Single point of entry for loading a data set accordingly to a previously chosen strategy.

        Args:
            data_set: A data set object containing reference to data.
            validation_size: A Numpy.float32 variable to define the percentage of the validation set size.
            nb_bins: In case of unsupervised discretization, represents the number of bins used in the previous process.

        Returns:
            A DataSet object containing both training and validation sets.
        """

        data_set = self._strategy.discretize(data_set=data_set, validation_size=validation_size, nb_bins=nb_bins)
        return data_set

    def set_strategy(self, strategy):
        """ Set a data set strategy. """
        self._strategy = strategy
