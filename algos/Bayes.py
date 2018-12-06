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

from helpers import utilities as Utils
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from tabulate import tabulate as Tabulate

class bayes(object):
    def __init__(self, validationMethod, bayesType, isDiscretize):
        """
        Initialisation d'un classifier Bayes
        Args:
            validationMethod:           Array de données
            bayesType:            Array de label
            isDiscretize:   Type de validation à utiliser
        """
        print("\nNew Bayes Classifier")
        self.validationMethod = validationMethod
        if bayesType == "multinomialnb":
            self.bayesType = MultinomialNB()
        elif bayesType == "gaussiannb":
            self.bayesType = GaussianNB()
        self.isDiscretize = isDiscretize
        self.precision = []
        self.score_f1 = []

    def train(self, features, answers):
        """
        Initialisation d'un classifier KNN
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Training \n")
        bayesPerf = [['Data type', 'Accuracy', 'Precision', 'F1']]
        params = dict()
        grid = GridSearchCV(self.bayesType,
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1,
                            iid=True,
                            scoring={'accuracy', 'precision', 'f1'},
                            refit='accuracy')

        #Fit data to Bayes algo
        dataType = 'Normalized'
        if self.isDiscretize:
            features = self.discritizeData(features)
            dataType = 'Discretized'
        grid.fit(features, answers)

        bayesPerf.append([dataType,
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_accuracy'])*100),
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_precision'])*100),
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_f1'])*100)])
        self.precision = grid.cv_results_['mean_test_accuracy']
        self.score_f1 = grid.cv_results_['mean_test_f1']

        print(Tabulate(bayesPerf, headers="firstrow"))
        print("-> Done\n\n")

    def discritizeData(self, features):
        """
        Discritization des données
        Args:
        Return:
            xKBinsDiscretizer: Array de données discritizé
        """
        est = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        est.fit(features)
        xKBinsDiscretizer = est.transform(features)
        return xKBinsDiscretizer

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
            Utils.printGraph('Type de données', 'Précision', [0.3], self.precision, precision)
            Utils.printGraph('Type de données', 'Score F1', [0.8], self.score_f1, score_f1)

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
