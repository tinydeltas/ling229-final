#!/usr/bin/python

import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc_curve(fpr, tpr, roc_auc, outfile="metrics/figs/roc_logit_figure.png"):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Logistic Regression Model on Post Data")
    plt.legend(loc="lower right")
    plt.savefig(outfile)
    plt.show()


def create_and_plot_roc_curve(y, scores, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc)


if __name__ == "__main__":
    label_location = sys.argv[1]
    probs_location = sys.argv[2]

    true_labels = np.loadtxt(label_location, dtype=int)
    probs = np.loadtxt(probs_location, dtype=float, delimiter=",")[: , 1]

    create_and_plot_roc_curve(true_labels, probs, pos_label=1)
