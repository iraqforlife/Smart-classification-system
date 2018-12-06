#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab 4 - Lab's name

Students :
Amhad Al-Taher — ALTA22109307
Jean-Philippe Decoste - DECJ19059105
Stéphanie Lacerte - LACS

Group :
GTI770-A18-02
"""

from helpers import utilities as Utils
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors  import KNeighborsClassifier
from tabulate import tabulate as Tabulate

class knn(object):
    def __init__(self, validationMethod, n_neighbors, weights):
        """
        Initialisation d'un classifier KNN
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("\nNew Knn Classifier")
        self.validationMethod = validationMethod
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.precision = []
        self.score_f1 = []
        self.best_params = []

    def evaluate(self, features, answers):
        """
        Initialisation d'un classifier KNN
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Training \n")
        knnPerf = [['Weights', 'K', 'Accuracy']]
        params = dict(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=['auto'])
        grid = GridSearchCV(KNeighborsClassifier(),
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=1,
                            iid=False)

        #Fit data to knn algo
        grid.fit(features, answers)

        #Loop through results and keep trace
        for i in range(0, len(grid.cv_results_['params'])):
            knnPerf.append([grid.cv_results_['params'][i]['weights'],
                            grid.cv_results_['params'][i]['n_neighbors'],
                            "{0:.2f}".format(grid.cv_results_['mean_test_score'][i]*100)])
               
            self.precision.append([grid.cv_results_['params'][i]['weights'], grid.cv_results_['mean_test_score'][i]])
            self.score_f1.append([grid.cv_results_['params'][i]['weights'], grid.cv_results_['mean_test_score'][i]])

        print(Tabulate(knnPerf, headers="firstrow"))
        print(grid.best_params_)
        print("\nThe best is KNN %s With K = %s" %(grid.best_params_['weights'], grid.best_params_['n_neighbors']))
        print()
        
        self.best_params = grid.best_params_
        print("-> Done\n\n")

        print(self.precision)

    def printGraph(self, precision, score_f1):
        """
        Affiche un graphique comparant la précision et le score F1 de deux classifier
        Args:
            precision: Array des données de précision du classifier en comparaison
            score_f1: Array des données de score F1 du classifier en comparaison
        """
        if (len(self.precision) != len(precision)):
            print("Les tableaux de données fournient n'ont pas la même dimension")
        else:
            Utils.printGraph('K', 'Précision', [1, 3, 5, 10], self.precision, precision)
            Utils.printGraph('K', 'Score F1', [1, 3, 5, 10], self.score_f1, score_f1)

    def getPrecision(self):
        """
        Renvoi un array des données de précision
        Args:

        Return: 
            precision: Array des données de précision du classifier en comparaison
        """
        return self.precision

    def getScoreF1(self):
        """
        Renvoi un array des données de score f1
        Args:

        Return:
            score_f1: Array des données de score F1 du classifier en comparaison
        """
        return self.score_f1

    def getBestParams(self):
        """
        Renvoi un array des meilleurs paramètres
        Args:

        Return:
            score_f1: Array des meilleurs paramètres du classifier en comparaison
        """
        return self.best_params
