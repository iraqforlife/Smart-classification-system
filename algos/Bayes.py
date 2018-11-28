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
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from tabulate import tabulate

VALIDATION_METHOD = ['Holdout', 'Stratified Shuffle Split']

def bayes(features, answers):
    validationCounter = 1
    for method in VALIDATION_METHOD:
        validation = None
        if method == 'Holdout':
            validation = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        elif method == 'Stratified Shuffle Split':
            validation = StratifiedShuffleSplit()
            
        print(str(validationCounter)+".Training with "+method+"\n")
        bayesPerf = [['Data type', 'Accuracy', 'Precision', 'F1']]
        params = dict()
        grid = GridSearchCV(MultinomialNB(), param_grid=params, cv=validation, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')

        #Scale the data between 0 and 1
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(features)
        xNormalized = scaler.transform(features)

        #Fit normalized data to Bayes algo
        grid.fit(xNormalized, answers)
        bayesPerf.append(['Normalized',
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_accuracy'])*100),
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_precision'])*100),
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_f1'])*100)])
        precision_norm = grid.cv_results_['mean_test_accuracy']
        score_f1_norm = grid.cv_results_['mean_test_f1']

        #Discretize normalized data
        est = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        est.fit(features)
        xKBinsDiscretizer = est.transform(features)

        #Fit discretize data to bayes algorithm
        grid.fit(xKBinsDiscretizer, answers)
        bayesPerf.append(['Discretized',
                            "{0:.2f}".format(float(grid.cv_results_['mean_test_accuracy'])*100),
                            "{0:.2f}".format(float(grid.cv_results_['mean_test_precision'])*100),
                            "{0:.2f}".format(float(grid.cv_results_['mean_test_f1'])*100)])
        precision_disct = grid.cv_results_['mean_test_accuracy']
        score_f1_disct = grid.cv_results_['mean_test_f1']

        print(tabulate(bayesPerf, headers="firstrow"))
        label = "Discritized"
        if precision_norm > precision_disct:
            label = "Normalized"
            print("\nThe best is Bayes with %s data" %(label))
        print()
        
        utils.printGraph('Type de données', 'Précision', [0.3], precision_norm, precision_disct)
        utils.printGraph('Type de données', 'Score F1', [0.8], score_f1_norm, score_f1_disct)
        validationCounter += 1
    
    print("\n"+str(validationCounter)+".Training with 10-fold cross-validation\n")
    bayesPerf = [['Data type', 'Accuracy', 'Precision', 'F1']]
    params = dict()
    bestGrid = GridSearchCV(MultinomialNB(), param_grid=params, cv=10, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')
    
    #Fit data to Bayes algo    
    bestGrid.fit(xNormalized, answers)
    bayesPerf.append(['Normalized',
                        "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_accuracy'])*100),
                        "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_precision'])*100),
                        "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_f1'])*100)])
    
    bestGrid.fit(xKBinsDiscretizer, answers)    
    bayesPerf.append(['Discretized',
                        "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_accuracy'])*100),
                        "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_precision'])*100),
                        "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_f1'])*100)])
    
    print(tabulate(bayesPerf, headers="firstrow"))
    print("-> Done\n\n")

def bayesGaussian(features, answers):
    validationCounter = 1
    for method in VALIDATION_METHOD:
        validation = None
        if method == 'Holdout':
            validation = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        elif method == 'Stratified Shuffle Split':
            validation = StratifiedShuffleSplit()
            
        print(str(validationCounter)+".Training with "+method+"\n")
        bayesPerf = [['Accuracy', 'Precision', 'F1']]
        params = dict()
        holdoutValidation = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        grid = GridSearchCV(GaussianNB(), param_grid=params, cv=validation, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')

        #Fit normalized data to Bayes algo
        grid.fit(features, answers)
        bayesPerf.append(["{0:.2f}".format(float(grid.cv_results_['mean_test_accuracy'])*100),
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_precision'])*100),
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_f1'])*100)])

        print(tabulate(bayesPerf, headers="firstrow"))
        print()
        
        validationCounter += 1
    
    print("\n"+str(validationCounter)+".Training with 10-fold cross-validation\n")
    bayesPerf = [['Accuracy', 'Precision', 'F1']]
    params = dict()
    bestGrid = GridSearchCV(GaussianNB(), param_grid=params, cv=10, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')
    
    #Fit data to Bayes algo
    bestGrid.fit(features, answers)
    
    bayesPerf.append(["{0:.2f}".format(float(bestGrid.cv_results_['mean_test_accuracy'])*100),
                      "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_precision'])*100),
                      "{0:.2f}".format(float(bestGrid.cv_results_['mean_test_f1'])*100)])
    
    print(tabulate(bayesPerf, headers="firstrow"))
    print("-> Done\n\n")