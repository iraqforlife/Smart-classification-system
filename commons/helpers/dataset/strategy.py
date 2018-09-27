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

import abc


class Strategy(metaclass=abc.ABCMeta):
    """
        Declare an interface common to all supported algorithms. Context uses this interface to call the algorithm
        defined by a concrete strategy.
    """

    @abc.abstractmethod
    def load_dataset(self):
        pass
