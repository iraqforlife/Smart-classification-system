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
Stéphanie Lacerte - LACS

Group :
GTI770-A18-02
"""

from helpers import utilities as utils
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tabulate import tabulate

VALIDATION_METHOD = ['Holdout', 'Stratified Shuffle Split']

def svm(features, answers, dataset_splitter):    
    #linear    
    print("SVM linear")
    c=[0.001,0.1,1.0,10.0]
    params = dict(kernel=['linear'], C=c, class_weight=['balanced'], cache_size=[2048])
    grid = GridSearchCV(SVC(), param_grid=params, cv=dataset_splitter, n_jobs=-1, iid=True)
    
    #Fit the feature to svm algo
    grid.fit(features, answers)
    
    #build table
    outPut = []
    y1 = []
    for i in range(0, 4):
        outPut.append([grid.cv_results_['params'][i]['C'],
                          "{0:.2f}%".format(grid.cv_results_['mean_test_score'][i]*100)])
        y1.append(grid.cv_results_['mean_test_score'][i]*100)
    
    #print table
    print(tabulate(outPut, headers=['Variable C', 'class_weight= {‘balanced’}']))
    print("The best parameters are ", grid.best_params_, " with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
    
    #rbf
    print("\nSVM rbf")
    params = dict(kernel=['rbf'], C=c, gamma=c ,class_weight=['balanced'], cache_size=[2048])
    grid = GridSearchCV(SVC(), param_grid=params, cv=dataset_splitter, n_jobs=-1, iid=True)
    
    #Fit the feature to svm algo
    grid.fit(features, answers)
    
    #build table
    outPut = []
    outPut.append(["0.001",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][0]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][1]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][2]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][3]*100)])
    outPut.append(["0.1",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][4]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][5]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][6]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][7]*100)])
    outPut.append(["1.0",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][8]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][9]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][10]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][11]*100)])
    outPut.append(["10.0",
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][12]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][13]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][14]*100),
                   "{0:.2f}%".format(grid.cv_results_['mean_test_score'][15]*100)])
    
    #print table
    print(tabulate(outPut, headers=['Variable C', 'a=0.001', 'a=0.1', 'a=1.0', 'a=10.0']))
    print("The best parameters are ", grid.best_params_, " with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
    
    print("-> Done\n\n")
