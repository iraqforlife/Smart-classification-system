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
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate as Tabulate

class decisionTree(object):
    def __init__(self, validationMethod, max_depth):
        """
        Initialisation d'un classifier KNN
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("\nNew Decision Tree Classifier")        
        self.validationMethod = validationMethod
        self.max_depth = max_depth
        self.precision = []
        self.score_f1 = []
        self.best_params = []

    def evaluate(self, features, answers):
        """
        Évaluation d'un classifier Decision Tree
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Evaluation \n")
        dTreePerf = [['Depth', 'Accuracy', 'Precision', 'F1']]
        params = dict(max_depth=self.max_depth)
        grid = GridSearchCV(DecisionTreeClassifier(),
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1,
                            iid=True,
                            scoring={'accuracy', 'precision', 'f1'},
                            refit='accuracy')

        #Fit data to Decision Tree algo
        grid.fit(features, answers)

        #Loop through results
        for i in range(0, 4):
            dTreePerf.append([grid.cv_results_['params'][i]['max_depth'],
                                "{0:.2f}".format(grid.cv_results_['mean_test_accuracy'][i]*100),
                                "{0:.2f}".format(grid.cv_results_['mean_test_precision'][i]*100),
                                "{0:.2f}".format(grid.cv_results_['mean_test_f1'][i]*100)])
            self.precision.append(grid.cv_results_['mean_test_accuracy'][i])
            self.score_f1.append(grid.cv_results_['mean_test_f1'][i])
            
        print(Tabulate(dTreePerf, headers="firstrow"))
        print("\nThe best is depth = %s" %(grid.best_params_['max_depth']))
        print()
        
        self.best_params = grid.best_params_
        print("-> Done\n\n")
    
    def train(self, features, answers):
        """
        Entrainement du meilleur modèle
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Training \n")
        dTreePerf = [['Depth', 'Accuracy']]
        model = DecisionTreeClassifier(max_depth=self.max_depth)

        #Fit data to Decision Tree algo
        model.fit(features, answers)

        #Save through results
        dTreePerf.append([self.max_depth, "{0:.2f}".format(model.score(features, answers)*100)])
        self.precision.append(model.score(features, answers))
            
        print(Tabulate(dTreePerf, headers="firstrow"))
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
            Utils.printGraph('Profondeur de l\'arbre', 'Précision', [0, 3, 5, 10], self.precision, precision)
            Utils.printGraph('Profondeur de l\'arbre', 'Score F1', [0, 3, 5, 10], self.score_f1, score_f1)

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
