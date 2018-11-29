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
        
        #transform classes into numbers
        if self.answer == 'BIG_BAND':
            self.answer = 0
        elif self.answer == 'BLUES_CONTEMPORARY':
            self.answer = 1
        elif self.answer == 'COUNTRY_TRADITIONAL':
            self.answer = 2
        elif self.answer == 'DANCE':
            self.answer = 3
        elif self.answer == 'ELECTRONICA':
            self.answer = 4
        elif self.answer == 'EXPERIMENTAL':
            self.answer = 5
        elif self.answer == 'FOLK_INTERNATIONAL':
            self.answer = 6
        elif self.answer == 'GOSPEL':
            self.answer = 7
        elif self.answer == 'GRUNGE_EMO':
            self.answer = 8
        elif self.answer == 'HIP_HOP_RAP':
            self.answer = 9
        elif self.answer == 'JAZZ_CLASSIC':
            self.answer = 10
        elif self.answer == 'METAL_ALTERNATIVE':
            self.answer = 11
        elif self.answer == 'METAL_DEATH':
            self.answer = 12
        elif self.answer == 'METAL_HEAVY':
            self.answer = 13
        elif self.answer == 'POP_CONTEMPORARY':
            self.answer = 14
        elif self.answer == 'POP_INDIE':
            self.answer = 15
        elif self.answer == 'POP_LATIN':
            self.answer = 16
        elif self.answer == 'PUNK':
            self.answer = 17
        elif self.answer == 'REGGAE':
            self.answer = 18
        elif self.answer == 'RNB_SOUL':
            self.answer = 19
        elif self.answer == 'ROCK_ALTERNATIVE':
            self.answer = 20
        elif self.answer == 'ROCK_COLLEGE':
            self.answer = 21
        elif self.answer == 'ROCK_CONTEMPORARY':
            self.answer = 22
        elif self.answer == 'ROCK_HARD':
            self.answer = 23
        elif self.answer == 'ROCK_NEO_PSYCHEDELIA':
            self.answer = 24