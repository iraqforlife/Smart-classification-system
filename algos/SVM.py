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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from tabulate import tabulate as Tabulate

class svm(object):
    def __init__(self, validationMethod, kernel, c, gamma, class_weight, cache_size):
        """
        Initialisation d'un classifier
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("\nNew SVM Classifier")
        self.validationMethod = validationMethod
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.precision = []
        self.score_f1 = []
        self.best_params = []
        self.best_score = 0
    
    def evaluate(self, features, answers):
        """
        Évaluation d'un classifier
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Evaluation")
        if self.gamma:
            print("--With gamma \n")
            svmPerf = [['Variable C', 'gamma', 'class_weight', 'Precision']]
            params = dict(kernel=self.kernel, C=self.c, gamma=self.gamma, class_weight=self.class_weight, cache_size=self.cache_size)
        else:
            print("--Without gamma \n")
            svmPerf = [['Variable C', 'class_weight', 'Precision']]
            params = dict(kernel=self.kernel, C=self.c, class_weight=self.class_weight, cache_size=self.cache_size)
        grid = GridSearchCV(SVC(),
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1)

        #Fit the feature
        grid.fit(features, answers)
        
        #Loop through results and keep trace
        for i in range(0, len(grid.cv_results_['params'])):
            if self.gamma:
                svmPerf.append([grid.cv_results_['params'][i]['C'],
                                grid.cv_results_['params'][i]['gamma'],
                                grid.cv_results_['params'][i]['class_weight'],
                                "{0:.2f}%".format(grid.cv_results_['mean_test_score'][i]*100)])
            else:
                svmPerf.append([grid.cv_results_['params'][i]['C'],
                                grid.cv_results_['params'][i]['class_weight'],
                                "{0:.2f}%".format(grid.cv_results_['mean_test_score'][i]*100)])
            self.precision.append(grid.cv_results_['mean_test_score'][i]*100)
            self.score_f1.append(grid.cv_results_['mean_test_score'][i]*100)
        
        #Print table
        print(Tabulate(svmPerf, headers='firstrow'))
        print("The best parameters are ", grid.best_params_, " with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
        print()

        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        print("-> Done\n\n")

    def train(self, features, answers):
        """
        Entrainement d'un classifier
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Training")
        if self.gamma:
            print("--With gamma \n")
            svmPerf = [['Variable C', 'gamma', 'class_weight', 'Precision']]
            params = dict(kernel=self.kernel, C=self.c, gamma=self.gamma, class_weight=self.class_weight, cache_size=self.cache_size)
        else:
            print("--Without gamma \n")
            svmPerf = [['Variable C', 'class_weight', 'Precision']]
            params = dict(kernel=self.kernel, C=self.c, class_weight=self.class_weight, cache_size=self.cache_size)

        grid = GridSearchCV(SVC(),
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1)

        #Fit the feature
        grid.fit(features, answers)
        
        #Loop through results and keep trace
        for i in range(0, len(grid.cv_results_['params'])):
            if self.gamma:
                svmPerf.append([grid.cv_results_['params'][i]['C'],
                                grid.cv_results_['params'][i]['gamma'],
                                grid.cv_results_['params'][i]['class_weight'],
                                "{0:.2f}%".format(grid.cv_results_['mean_test_score'][i]*100)])
            else:
                svmPerf.append([grid.cv_results_['params'][i]['C'],
                                grid.cv_results_['params'][i]['class_weight'],
                                "{0:.2f}%".format(grid.cv_results_['mean_test_score'][i]*100)])
            self.precision.append(grid.cv_results_['mean_test_score'][i]*100)
            self.score_f1.append(grid.cv_results_['mean_test_score'][i]*100)
        
        #print table
        print(Tabulate(svmPerf, headers='firstrow'))
        print()
        
        print("-> Done\n\n")

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

    def getDefinition(self):
        """
        Renvoi la définition du modèle
        Args:

        Return:
            best_score: Le meilleur résultat
        """
        return ['Machine à vecteur(SVM):', self.best_params]

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
