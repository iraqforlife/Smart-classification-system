#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab 3

Students :
Amhad Al-Taher — Permanent Code
Jean-Philippe Decoste - DECJ19059105

Group :
GTI770-A18-02
"""

class Image:    
    def __init__(self, features):
        self.features = []
        
        self.answer = features[len(features)-1] == 1
        self.id = int(features[0]) 
                
        self.features = []
        #Couleur moyenne [0]
        self.features.append(features[4])
        #Couleur moyenne [1]
        self.features.append(features[5])
        #Couleur moyenne [2]
        self.features.append(features[6])
        #Excentricité
        self.features.append(features[19])
        #Entropie
        self.features.append(features[23])
        #Chiralité
        self.features.append(features[24])
        