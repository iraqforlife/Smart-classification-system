#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab 1 - Lab's name

Students :
Amhad Al-Taher — Permanent Code
Jean-Philippe Decoste - DECJ19059105

Group :
GTI770-A18-02
"""

class Spam:
    def __init__(self, features):
        self.features = features
        self.answer = int(self.features[len(self.features)-1]) == 1
        del self.features[-1]
