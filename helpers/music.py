#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab 4

Students :
Amhad Al-Taher — Permanent Code
Jean-Philippe Decoste - DECJ19059105

Group :
GTI770-A18-02
"""

class Music:    
    def __init__(self, features):
        
        self.answer = features[-1]
        self.id = features[1]
        #remove the answer
        del features[-1]
        #remove the identifier (ex:27)
        del features[0]
        #remove The id (ex:TRAAAAK128F9318786)
        del features[0]
        
        self.features = [float(i) for i in features]