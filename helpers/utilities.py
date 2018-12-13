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

from algos import DecisionTree
from algos import Knn
from algos import Bayes
from algos import NeuralNetwork
from algos import SVM
import matplotlib.pyplot as plt

def printProgressBar(iteration, total, prefix='', suffix='', length=50, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')


def getBestModel(models):
    """
    Compare deux modèle différent du classifier et retourne le meilleur
    Args:
        models: Liste des Models en comparaison
    """
    best_model = None

    for model in models:
        if best_model is None:
            best_model = model
        else:
            if best_model.getBestScore() < model.getBestScore():
                best_model = model

    return best_model

def findBaselineClassifier(algos, features, answers):
    print("Train following algo to find baseLine: ")
    print(algos)
    comparison = []

    if 'decisionTree' in algos:
        # Decision Tree
        basicDTree = DecisionTree.decisionTree(5, [5, 10, 15])
        basicDTree.evaluate(features, answers)
        # Get Best
        params = basicDTree.getBestParams()
        bestDTree = DecisionTree.decisionTree(10, params['max_depth'])
        bestDTree.train(features, answers)
        #---------------------------------------------------------------------------------------
        comparison.append(bestDTree)
    
    if 'knn' in algos:
        # KNN
        basicKnn = Knn.knn(5, [5, 10, 20], ['distance'])
        basicKnn.evaluate(features, answers)
        # Get Best
        params = basicKnn.getBestParams()
        bestKnn = Knn.knn(10, params['n_neighbors'], params['weights'])
        bestKnn.train(features, answers)
        #---------------------------------------------------------------------------------------
        comparison.append(bestKnn)

    if 'bayes' in algos:
        # Bayes
        # Multinomial Non-Discritized
        #basicBayesM = Bayes.bayes(5, "multinomialnb", False)
        #basicBayesM.evaluate(features, answers)
        # WHY: Valeur négative cause des problèmes
        # Multinomial Discritized
        #basicBayesM2 = Bayes.bayes(5, "gaussiannb", True)
        #basicBayesM2.evaluate(features, answers)
        # WHY: score de 3%
        # Gaussian
        #basicBayesG = Bayes.bayes(5, "gaussiannb", False)
        #basicBayesG.evaluate(features, answers)
        # Get Best
        bestBayes = Bayes.bayes(10, "gaussiannb", False)
        bestBayes.train(features, answers)
        #---------------------------------------------------------------------------------------
        comparison.append(bestBayes)

    if 'neuralNetwork' in algos:
        # Neural Network
        # Two layers
        """
        twoLayerMLP = NeuralNetwork.neuralNetwork(2, 1,
                                                [25],
                                                0.001,
                                                len(features[0]),
                                                ['relu', 'softmax'],
                                                [10, 100],
                                                [500, 1000])
        twoLayerMLP.evaluate(features, answers)
        WHY : .02% de diff
        """
        # Four layers
        fourLayerMLP = NeuralNetwork.neuralNetwork(2, 2,
                                                [25, 30, 15],
                                                0.01,
                                                len(features[0]),
                                                ['relu', 'softmax'],
                                                [250, 800],
                                                [1000, 5000])
        fourLayerMLP.evaluate(features, answers)
        # Get Best
        params = fourLayerMLP.getBestParams()
        bestMLP = NeuralNetwork.neuralNetwork(5, 3,
                                                params['layers'],
                                                params['learn_rate'],
                                                len(features[0]),
                                                params['activation'],
                                                params['epochs'],
                                                params['batch_size'])
        bestMLP.train(features, answers)
        #---------------------------------------------------------------------------------------
        comparison.append(bestMLP)

    if 'SVM' in algos:
        # SVM
        # Linear
        linearSVM = SVM.svm(2, ['linear'], [0.1, 1.0, 5.0], [], ['balanced'], [2048])
        linearSVM.evaluate(features, answers)
        # RBF
        rbfSVM = SVM.svm(2, ['rbf'], [0.1, 1.0, 5.0], [0.1, 1.0, 5.0], ['balanced'], [2048])
        rbfSVM.evaluate(features, answers)
        # Get Best
        temp = getBestModel([linearSVM, rbfSVM])
        params = temp.getBestParams()
        bestSVM = SVM.svm(5, params['kernel'], params['C'], params['gamma'], ['balanced'], [2048])
        bestSVM.train(features, answers)
        #---------------------------------------------------------------------------------------
        comparison.append(bestSVM)

    # Find baseline classifier
    return getBestModel(comparison)

def printGraph(xLabel, yLabel, xValues, yValues, yValues2):
    plt.grid(True) 
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    minY = min(yValues)
    maxY = max(yValues)

    if len(xValues) > 1:
        plt.plot(xValues, yValues, 'b')
    else:
        plt.plot(0.3, yValues, 'bs')
    if yValues2:
        if len(xValues) > 1:
            plt.plot(xValues, yValues2, 'r')
        else:
            plt.plot(0.8, yValues2, 'r^')
        
        if min(yValues) > min(yValues2):
            minY = min(yValues2)
        if max(yValues) < max(yValues2):
            maxY = max(yValues2)

    if len(xValues) > 1:
        plt.xlim(min(xValues), max(xValues))
    else:
        plt.xlim(0, 1)
    plt.ylim(minY-.025, maxY+.025)

    plt.show()