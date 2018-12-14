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
        self.best_score = 0

    def evaluate(self, features, answers):
        """
        Évaluation d'un classifier
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Evaluation \n")
        knnPerf = [['Weights', 'K', 'Accuracy']]
        params = dict(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=['auto'])
        grid = GridSearchCV(KNeighborsClassifier(),
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1)

        #Fit data to knn algo
        grid.fit(features, answers)

        #Loop through results and keep trace
        for i in range(0, len(grid.cv_results_['params'])):
            knnPerf.append([grid.cv_results_['params'][i]['weights'],
                            grid.cv_results_['params'][i]['n_neighbors'],
                            "{0:.2f}".format(grid.cv_results_['mean_test_score'][i]*100)])
               
            self.precision.append([grid.cv_results_['params'][i]['weights'], grid.cv_results_['mean_test_score'][i]])
            self.score_f1.append([grid.cv_results_['params'][i]['weights'], grid.cv_results_['mean_test_score'][i]])

        #Print table
        print(Tabulate(knnPerf, headers="firstrow"))
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
        print("1.Training \n")
        knnPerf = [['Weights', 'K', 'Accuracy']]
        model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)

        #Fit data to Decision Tree algo
        model.fit(features, answers)

        #Save results
        knnPerf.append([self.weights, self.n_neighbors, "{0:.2f}".format(model.score(features, answers)*100)])
        self.precision.append(model.score(features, answers))
        self.best_score = self.precision[0]

        print(Tabulate(knnPerf, headers="firstrow"))
        print()

        print("-> Done\n\n")
        return model

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
        return ['Plus proche voisin (KNN):', self.best_params]

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

    def getBestScore(self):
        """
        Renvoi le meilleur score
        Args:

        Return:
            best_score: Le meilleur résultat
        """
        return self.best_score
