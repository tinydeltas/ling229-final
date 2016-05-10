#!/usr/bin/python

import sys
import numpy as np
import pylab
from sklearn.decomposition import PCA


def reduce_dim(data, ndim=2):
    pca_model = PCA(n_components=ndim)
    reduced_data = pca_model.fit_transform(data)
    return reduced_data


def plot_reduced_data(data, labels, outfile="metrics/figs/pca_plotted_data.png"):
    colors = ["green" if label else "red" for label in labels]

    pylab.scatter(data[:, 0], data[:, 1], c=colors)
    pylab.ylim(-120,40)
    pylab.xlim(-150,1000)
    pylab.ylabel("Principal Component 2")
    pylab.xlabel("Principal Component 1")

    pylab.title("PCA-Reduced Post Features")

    if outfile is not None:
        pylab.savefig(outfile)

    pylab.show()

if __name__ == '__main__':
    feat_file = sys.argv[1]
    label_file = sys.argv[2]
    feats = np.loadtxt(feat_file, dtype=float, delimiter=",")
    labels = np.loadtxt(label_file, dtype=int, delimiter=",")

    random_ordering = np.arange(0, len(labels))
    np.random.shuffle(random_ordering)

    feats = feats[random_ordering, :]
    labels = labels[random_ordering]

    reduced_data = reduce_dim(feats, 2)
    plot_reduced_data(reduced_data, labels)
