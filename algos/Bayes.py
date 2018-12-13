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
            validationMethod:   Array de données
            bayesType:          Array de label
            isDiscretize:       Type de validation à utiliser
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
        self.best_params = []
        self.best_score = 0

    def evaluate(self, features, answers):
        """
        Évaluation d'un classifier Bayes
        Args:
            features:   Array de données
            answers:    Array de label
        """
        print("1.Evaluation \n")
        bayesPerf = [['Data type', 'Accuracy']]
        params = dict()
        grid = GridSearchCV(self.bayesType,
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1)

        #Fit data to Bayes algo
        dataType = 'Normalized'
        if self.isDiscretize:
            features = self.discritizeData(features)
            dataType = 'Discretized'
        grid.fit(features, answers)

        #Loop through results and keep trace
        bayesPerf.append([dataType,
                          "{0:.2f}".format(float(grid.cv_results_['mean_test_score'])*100)])
        self.precision = grid.cv_results_['mean_test_score']
        self.score_f1 = grid.cv_results_['mean_test_score']

        #print table
        print(Tabulate(bayesPerf, headers="firstrow"))
        print("The best parameters are ", grid.best_params_, 
              " with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
        print()

        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        print("-> Done\n\n")

    def train(self, features, answers):
        """
        Entrainement d'un classifier Bayes
        Args:
            features:   Array de données
            answers:    Array de label
        """
        print("1.Training \n")
        bayesPerf = [['Data type', 'Accuracy']]
        model = self.bayesType

        #Fit data to Decision Tree algo
        model.fit(features, answers)

        #Save results
        bayesPerf.append(['Normalized', "{0:.2f}".format(model.score(features, answers)*100)])
        self.precision.append(model.score(features, answers))
        self.best_score = self.precision[0]

        print(Tabulate(bayesPerf, headers="firstrow"))
        print()
        
        print("-> Done\n\n")

    def discritizeData(self, features):
        """
        Discritization des données
        Args:
            features:           Array de données
        Return:
            xKBinsDiscretizer:  Array de données discritizé
        """
        est = preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        est.fit(features)
        xKBinsDiscretizer = est.transform(features)
        return xKBinsDiscretizer

    def printGraph(self, precision, score_f1):
        """
        Affiche un graphique comparant la précision et le score F1 de deux classifier
        Args:
            precision:  Array des données de précision du classifier en comparaison
            score_f1:   Array des données de score F1 du classifier en comparaison
        """
        if (len(self.precision) != len(precision)):
            print("Les tableaux de données fournient n'ont pas la même dimension")
        else:
            Utils.printGraph('Type de données', 'Précision', [0.3], self.precision, precision)
            Utils.printGraph('Type de données', 'Score F1', [0.8], self.score_f1, score_f1)

    def getDefinition(self):
        """
        Renvoi la définition du modèle
        Args:

        Return:
            best_score: Le meilleur résultat
        """
        return ['Bayes:', self.best_params]

    def getPrecision(self):
        """
        Renvoi un array des données de précision
        Args:

        Return:
            precision:  Array des données de précision du classifier en comparaison
        """
        return self.precision

    def getScoreF1(self):
        """
        Renvoi un array des données de score f1
        Args:

        Return:
            score_f1:   Array des données de score F1 du classifier en comparaison
        """
        return self.score_f1

    def getBestParams(self):
        """
        Renvoi un array des meilleurs paramètres
        Args:

        Return:
            best_params: Array des meilleurs paramètres du classifier en comparaison
        """
        return self.best_params

    def getBestScore(self):
        """
        Renvoi le meilleur score
        Args:

        Return:
            best_score: Le meilleur résultat
        """
        return self.best_score
