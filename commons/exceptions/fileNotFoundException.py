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

class FileNotFoundException(Exception):
    """ Custom exception for FileNotFound error. """
    def __init__(self, message):
        self.message = message
