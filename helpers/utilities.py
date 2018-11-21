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

import matplotlib.pyplot as plt

#LAB01
IMAGE_CSV_NAME = r"data\csv\galaxy\galaxy_label_data_set.csv"
IMAGETEST_CSV_NAME = r"data\csv\galaxy\galaxy_label_data_set_test.csv"

#Galaxy features
EXTRATED_GALAXY_PRIMITIVE = r"data\csv\eq07_pExtraction.csv"
MERGED_GALAXY_PRIMITIVE = r"data\csv\eq07_pMerged.csv"
ALL_GALAXY_PRIMITIVE = r"data\csv\galaxy\galaxy_feature_vectors.csv"

#Spam features
ALL_SPAM_PRIMITIVE = r"data\csv\spam\spam.csv"


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