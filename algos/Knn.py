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
from sklearn.neighbors  import KNeighborsClassifier
from tabulate import tabulate

# Params
N_NEIGHBORS = [1, 3, 5, 10, 15]
WEIGHTS = ['uniform', 'distance']

def knn(features, answers, doPrintGraph):
    """
    Algorithme 'KNN' utilisé pour classer les données qui lui sont fourni
    Args:
        --
    """
    validation = StratifiedShuffleSplit()
        
    print("1.Training \n")
    knnPerf = [['Weights', 'K', 'Accuracy', 'Precision', 'F1']]
    params = dict(n_neighbors=N_NEIGHBORS, weights=WEIGHTS, algorithm=['auto'])
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=validation, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')

    #Fit data to knn algo
    grid.fit(features, answers)

    #Loop through results
    precision_un = []
    score_f1_un = []
    precision_di = []
    score_f1_di = []
    for i in range(0, 6):
        knnPerf.append([grid.cv_results_['params'][i]['weights'],
                        grid.cv_results_['params'][i]['n_neighbors'],
                        "{0:.2f}".format(grid.cv_results_['mean_test_accuracy'][i]*100),
                        "{0:.2f}".format(grid.cv_results_['mean_test_precision'][i]*100),
                        "{0:.2f}".format(grid.cv_results_['mean_test_f1'][i]*100)])
        if grid.cv_results_['params'][i]['weights'] == 'uniform':
            precision_un.append(grid.cv_results_['mean_test_accuracy'][i])
            score_f1_un.append(grid.cv_results_['mean_test_f1'][i])
        elif grid.cv_results_['params'][i]['weights'] == 'distance':
            precision_di.append(grid.cv_results_['mean_test_accuracy'][i])
            score_f1_di.append(grid.cv_results_['mean_test_f1'][i])

    print(tabulate(knnPerf, headers="firstrow"))
    print("\nThe best is KNN %s With K = %s" %(grid.best_params_['weights'], grid.best_params_['n_neighbors']))
    print()
    
    if doPrintGraph:
        utils.printGraph('K', 'Précision', [3, 5, 10], precision_un, precision_di)
        utils.printGraph('K', 'Score F1', [3, 5, 10], score_f1_un, score_f1_di)
    
    print("\n2.Training best params with 10-fold cross-validation\n")
    knnPerf = [['Weights', 'K', 'Accuracy', 'Precision', 'F1']]
    params = dict(n_neighbors=[grid.best_params_['n_neighbors']], weights=[grid.best_params_['weights']], algorithm=['auto'])
    bestGrid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=10, n_jobs=-1, iid=True, scoring={'accuracy', 'precision', 'f1'}, refit='accuracy')
    
    #Fit data to knn algo
    bestGrid.fit(features, answers)
    
    knnPerf.append([bestGrid.cv_results_['params'][0]['weights'],
                    bestGrid.cv_results_['params'][0]['n_neighbors'],
                    "{0:.2f}".format(bestGrid.cv_results_['mean_test_accuracy'][0]*100),
                    "{0:.2f}".format(bestGrid.cv_results_['mean_test_precision'][0]*100),
                    "{0:.2f}".format(bestGrid.cv_results_['mean_test_f1'][0]*100)])
    
    print(tabulate(knnPerf, headers="firstrow"))
    print("-> Done\n\n")
