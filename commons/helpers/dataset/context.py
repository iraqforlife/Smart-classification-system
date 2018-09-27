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

class Context:
    """
        Define the interface of interest to clients. Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def load_dataset(self, csv_file, one_hot, validation_size):
        """ Single point of entry for loading a data set accordingly to a previously chosen strategy.

        Args:
            csv_file: A string representing the path the the reference data set file.
            one_hot: A booleans to loads the labels as one-hot vectors or not.
            validation_size: A Numpy.float32 variable to define the percentage of the validation set size.

        Returns:
            A DataSet object containing both training and validation sets.
        """
        data_set = self._strategy.load_dataset(csv_file, one_hot, validation_size)
        return data_set

    def set_strategy(self, strategy):
        """ Set a data set strategy. """
        self._strategy = strategy
