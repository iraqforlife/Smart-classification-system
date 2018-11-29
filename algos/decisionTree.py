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
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

TREE_DEPTH = [None, 3, 5, 10]
VALIDATION_METHOD = ['Holdout', 'Stratified Shuffle Split']

def decisionTree(features, answers, doPrintGraph):
    """
    Algorithme 'Decision Tree' utilisé pour classer les données qui lui sont fourni
    Args:
        --
    """
    validationCounter = 1
    for method in VALIDATION_METHOD:
        validation = None
        if method == 'Holdout':
            validation = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        elif method == 'Stratified Shuffle Split':
            validation = StratifiedShuffleSplit()
        
        print(str(validationCounter)+".Training with "+method+"\n")
        dTreePerf = [['Depth', 'Accuracy', 'Precision', 'F1']]
        params = dict(max_depth=TREE_DEPTH)
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=validation, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')

        #Fit data to Decision Tree algo
        grid.fit(features, answers)

        #Loop through results
        precision = []
        score_f1 = []
        for i in range(0, 4):
            dTreePerf.append([grid.cv_results_['params'][i]['max_depth'],
                              "{0:.2f}".format(grid.cv_results_['mean_test_accuracy'][i]*100),
                              "{0:.2f}".format(grid.cv_results_['mean_test_precision'][i]*100),
                              "{0:.2f}".format(grid.cv_results_['mean_test_f1'][i]*100)])
            precision.append(grid.cv_results_['mean_test_accuracy'][i])
            score_f1.append(grid.cv_results_['mean_test_f1'][i])
            
        print(tabulate(dTreePerf, headers="firstrow"))
        print("\nThe best is depth = %s" %(grid.best_params_['max_depth']))
        print()
        
        if doPrintGraph:
            utils.printGraph('Profondeur de l\'arbre', 'Précision', [0, 3, 5, 10], precision, [])
            utils.printGraph('Profondeur de l\'arbre', 'Score F1', [0, 3, 5, 10], score_f1, [])
        validationCounter += 1
    
    print(str(validationCounter)+".Training best params with 10-fold cross-validation\n")
    dTreePerf = [['Depth', 'Accuracy', 'Precision', 'F1']]
    params = dict(max_depth=[grid.best_params_['max_depth']])
    bestGrid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=10, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')
    
    #Fit data to Decision Tree algo
    bestGrid.fit(features, answers)
    
    dTreePerf.append([bestGrid.cv_results_['params'][0]['max_depth'],
                      "{0:.2f}".format(bestGrid.cv_results_['mean_test_accuracy'][0]*100),
                      "{0:.2f}".format(bestGrid.cv_results_['mean_test_precision'][0]*100),
                      "{0:.2f}".format(bestGrid.cv_results_['mean_test_f1'][0]*100)])
    
    print(tabulate(dTreePerf, headers="firstrow"))
    print("-> Done\n\n")
