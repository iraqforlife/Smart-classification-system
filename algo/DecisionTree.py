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

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

def decisionTree():
    """
    This method is used to build the decision tree and how the graph associated with it.
    fit(x,y) takes two arrays:
    X, sparse or dense, of size [n_samples, n_features] holding the training samples, and an array
            ex: [[0, 0], [1, 1]]
    Y of integer values, size [n_samples], holding the class labels
            ex : [0, 1]

    Since out Y array is not numerical we are using preprocessing from klearn to transform them
    Args:
        data: the array containing all the features. It will be used as the X
        labels: the array containing all the labels. It will be used as the Y
    """
    print("1.Training\n")
    dTreePerf = [['Depth', 'Score']]
    params = dict(max_depth=TREE_DEPTH)
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5, n_jobs=-1, verbose=50)

    #Fit data to Decision Tree algo
    grid.fit(features_train, answers_train)

    #Loop through results
    for i in range(0, 4):
        dTreePerf.append([grid.cv_results_['params'][i]['max_depth'],
                        "{0:.2f}".format(grid.cv_results_['mean_test_score'][i]*100)])

    print(tabulate(dTreePerf, headers="firstrow"))
    print("\nThe best is depth = %s" %(grid.best_params_['max_depth']))

    print("\n2.Validating performance with test data\n")
    answer_true, answer_pred = answers_test, grid.predict(features_test)
    print(classification_report(answer_true, answer_pred))