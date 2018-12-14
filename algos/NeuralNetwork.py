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
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate as Tabulate
import tensorflow as tf
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

TENSORBOARD_SUMMARY = r"tensorboard"

class neuralNetwork(object):
    def __init__(self, validationMethod, id, layers, learn_rate, dimension, activation, epoch, batch_size):
        """
        Initialisation d'un classifier MLP
        Args:
            validationMethod:   Méthode de validation
            id:                 Identifiant unique du modèle (utiliser pour les dossier tensorboard)
            layers:             Array contenant le squelette du réseau
            learn_rate:         Taux d'apprentissage (entre 0 et 1)
            dimension:          Dimension du dataset (détermine le nombre de perceptron de la couche 1)
            activation:         Array des méthodes d'activations (max=2, la deuxième est l'activation de la dernière couche)
            epoch:              Nombre d'itération
            batch_size:         Taille de chaque lot de données
        """
        print("\nNew Neural Network Classifier")
        self.validationMethod = validationMethod
        self.id = id
        self.layers = layers
        self.learn_rate = learn_rate
        self.dimension = dimension
        self.activation = activation
        self.epoch = epoch
        self.batch_size = batch_size
        self.precision = []
        self.score_f1 = []
        self.best_params = []
        self.best_score = 0

        """
        # Create a default in-process session.
        directory = TENSORBOARD_SUMMARY + "/" + str(id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("TensorBoard summary writer at :" + directory + "\n")
        self.tbCallBack = keras.callbacks.TensorBoard(log_dir=directory, histogram_freq=0, write_graph=False, write_images=False)
        """

    def create_model(self):
        """
        Crée un model MLP
        Args:

        """
        # The network type
        model = keras.Sequential()

        # Static Params
        counter = 1
        kernel_initializer = 'uniform'
        bias_initializer = 'zeros'
        dropout_rate = 0.1

        # Set layers
        # First layer is set according to data dimension
        model.add(keras.layers.Dense(self.dimension + 10,
                                     input_dim=self.dimension,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     activation=self.activation[0]))
        model.add(keras.layers.Dropout(dropout_rate))
        # Other layer set using layers array
        for perceptron in self.layers:
            if len(self.layers) == counter:
                # Last layer (25 neurons for 25 possible class, SIGMOID ensure result between 1 and 0)
                model.add(keras.layers.Dense(25,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             activation=self.activation[1]))
            else:
                # Adds Layer
                model.add(keras.layers.Dense(perceptron,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             activation=self.activation[0]))
                model.add(keras.layers.Dropout(dropout_rate))
            counter = counter + 1

        # Compile the network according to previous settings
        model.compile(optimizer=tf.train.AdamOptimizer(self.learn_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def evaluate(self, features, answers):
        """
        Évaluation d'un classifier
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Evaluation\n")

        mlpPerf = [['Epoch', 'Batch Size', 'Accuracy']]
        params = dict(epochs=self.epoch, batch_size=self.batch_size)
        grid = GridSearchCV(KerasClassifier(build_fn=self.create_model, verbose=0),
                            param_grid=params,
                            cv=self.validationMethod,
                            n_jobs=-1,
                            verbose=0)

        # Fix answer array
        answers = np_utils.to_categorical(answers)

        #Fit data to algo
        grid.fit(features, answers)

        #Loop through results
        for i in range(0, len(grid.cv_results_['params'])):
            mlpPerf.append([grid.cv_results_['params'][i]['epochs'],
                            grid.cv_results_['params'][i]['batch_size'],
                            "{0:.2f}".format(grid.cv_results_['mean_test_score'][i]*100)])
            self.precision.append(grid.cv_results_['mean_test_score'][i])
            self.score_f1.append(grid.cv_results_['mean_test_score'][i])

        self.best_params = grid.best_params_
        self.best_params['layers'] = self.layers
        self.best_params['learn_rate'] = self.learn_rate
        self.best_params['activation'] = self.activation
        self.best_params['layers'] = self.layers
        self.best_score = grid.best_score_

        #Print table
        print(Tabulate(mlpPerf, headers='firstrow'))
        print("The best parameters are ", self.best_params, 
              " with a score of {0:.2f}%".format(float(grid.best_score_)* 100))
        print()

        print("-> Done\n\n")

    def train(self, features, answers):
        """
        Entrainement du classifier MLP
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """
        print("1.Training")

        mlpPerf = [['Epoch', 'Batch Size', 'Accuracy']]
        model = KerasClassifier(build_fn=self.create_model, verbose=0)

        # Fix answer array
        answers = np_utils.to_categorical(answers)

        #Fit data to algo
        model.fit(features, answers)

        #Save results
        mlpPerf.append([self.epoch, self.batch_size, "{0:.2f}".format(model.score(features, answers)*100)])
        self.precision.append(model.score(features, answers))
        self.best_score = self.precision[0]
        
        #Print table
        print(Tabulate(mlpPerf, headers='firstrow'))
        print()
        
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
            Utils.printGraph('Profondeur de l\'arbre', 'Précision', [0, 3, 5, 10], self.precision, precision)
            Utils.printGraph('Profondeur de l\'arbre', 'Score F1', [0, 3, 5, 10], self.score_f1, score_f1)

    def getDefinition(self):
        """
        Renvoi la définition du modèle
        Args:

        Return:
            best_score: Le meilleur résultat
        """
        return ['Réseau de neurones:', self.best_params]

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
