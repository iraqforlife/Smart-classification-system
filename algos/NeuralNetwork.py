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
import tensorflow as tf
from tensorflow import keras
from tabulate import tabulate

TENSORBOARD_SUMMARY = r"tensorboard"

class neuralNetwork(object):
    def __init__(self, validationMethod, id, layers, epoch, learning_rate, dimension, batch_size, activations):
        """
        Initialisation d'un classifier MLP
        Args:
            validationMethod:   Méthode de validation
            id:                 Identifiant unique du modèle (utiliser pour les dossier tensorboard)
            networkFrame:       Array contenant le squelette du réseau
            epoch:              Nombre d'itération
            learning_rate:      Taux d'apprentissage (entre 0 et 1)
            dimension:          Dimension du dataset (détermine le nombre de perceptron de la couche 1)
            batch_size:         Taille de chaque lot de données
            activations:        Array des méthodes d'activations (max=2, la deuxième est l'activation de la dernière couche)
        """
        print("\nNew Neural Network Classifier")
        self.validationMethod = validationMethod
        self.id = id
        self.layers = layers
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.precision = []
        self.score_f1 = []

        # Create a default in-process session.
        directory = TENSORBOARD_SUMMARY + "/" + str(id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("TensorBoard summary writer at :" + directory + "\n")
        self.tbCallBack = keras.callbacks.TensorBoard(log_dir=directory, histogram_freq=0, write_graph=False, write_images=False)
        
        # The network type
        model = keras.Sequential()
        counter = 1

        # Set layer in model
        # First layer is set according to data dimension
        model.add(keras.layers.Dense(dimension, input_dim=dimension, kernel_initializer='random_normal', bias_initializer='zeros', activation=activations[0]))
        model.add(keras.layers.Dropout(0.2))
        # Other layer set using layers array
        for perceptron in layers:
            if len(layers) == counter:
                # Last layer (2 neurons for 2 possible class, SIGMOID ensure result between 1 and 0)
                if len(activations) > 1:
                    model.add(keras.layers.Dense(1, kernel_initializer='random_normal', bias_initializer='zeros', activation=activations[1]))
                else:
                    model.add(keras.layers.Dense(1, kernel_initializer='random_normal', bias_initializer='zeros', activation=activations[0]))
            else:
                # Adds Layer
                model.add(keras.layers.Dense(perceptron, kernel_initializer='random_normal', bias_initializer='zeros', activation=activations[0]))
                model.add(keras.layers.Dropout(0.2))
            counter = counter + 1

        self.model = model

    def train(self, features_train, answers_train, features_test, answers_test):
        """
        Entrainement du classifier MLP
        Args:
            features:           Array de données
            answers:            Array de label
            ValidationMethod:   Type de validation à utiliser
        """

        # Compile the network according to previous settings
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.learning_rate), 
                                    loss='binary_crossentropy', 
                                    metrics=['accuracy'])

        # Print visualisation of network (layer and perceptron)
        self.model.summary()

        # Fit model to data
        print("\n2.Training\n")
        self.model.fit(np.array(features_train), np.array(answers_train),
                                epochs=self.epoch,
                                batch_size=self.batch_size,
                                validation_data=(np.array(features_test), np.array(answers_test)),
                                callbacks=[self.tbCallBack],
                                verbose=2)

        # Evaluation
        #scores = neuralNetwork_model.evaluate(np.array(features_train), np.array(answers_train), verbose=1)
        #print("\n%s: %.2f%%" % (neuralNetwork_model.metrics_names[1], scores[1]*100))

        # Clear previous model
        keras.backend.clear_session()

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
