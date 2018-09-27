#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # X - Lab's name

Students :
    Names — Permanent Code

Group :
    GTI770-H18-0X
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classifiers.galaxy_classifiers.decision_tree_classifier import TreeClassifier


class Plot(object):
    """ Class to plot graphics. """

    @staticmethod
    def plot_feature_comparison(feature1, feature2, class_labels, filename="/tmp/feature_comparison.png"):
        """ Plot a graphics to evaluate features.

        Args:
            feature1: A numpy array of values for the first feature to evaluate.
            feature2: A numpy array of values for the second feature to evaluate.
            class_labels: The labels associated to the feature vector.
            filename: The desired file name to save the plot.

        Returns:
            None, but save plot to a file.
        """
        x = feature1
        y = feature2

        # Set graphics properties
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_title("Feature comparison", fontsize=14)
        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.grid(True, linestyle='-', color='0.75')

        # Wrap X, Y and labels.
        df = pd.DataFrame(dict(x=x, y=y, label=class_labels))
        groups = df.groupby('label')

        # Plot
        ax.margins(0.05)
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
        ax.legend()
        plt.savefig(filename)

    @staticmethod
    def plot_tree_decision_surface(X, y, feature_name1, feature_name2, filename="/tmp/decision_tree.png"):
        """ Plot the surface of decision of a decision tree.

        Args:
            X : the feature vector.
            y : ground truth labels.
            feature_name1 : a string defining the name of the first feature.
            feature_name2 : a string defining the name of the second feature.
            filename : path where the plot is saved.

        Returns:
            None, but saves plot to a file.
        """
        # Plot parameters.
        n_classes = 3
        plot_colors = "ryb"
        plot_step = 0.02
        target_names = ["smooth", "spiral", "artifact"]

        for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                        [1, 2], [1, 3], [2, 3]]):

            # Arranges the data in pair of features.
            x_features_train = X[:, pair]

            # Train a decision tree.
            clf = TreeClassifier()
            clf.train(X=x_features_train, y=y)

            # Plot the decision boundary
            plt.subplot(2, 3, pairidx + 1)

            x_min, x_max = x_features_train[:, 0].min() - 1, x_features_train[:, 0].max() + 1
            y_min, y_max = x_features_train[:, 1].min() - 1, x_features_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

            plt.xlabel(feature_name1)
            plt.ylabel(feature_name2)

            # Plot the training points
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(y == i)
                plt.scatter(x_features_train[idx, 0], x_features_train[idx, 1], c=color, label=target_names[i],
                            cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

        plt.suptitle("Decision surface of a decision tree using paired features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.savefig(filename)
