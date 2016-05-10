#!/usr/bin/python

import os
import sys

# hack to make this able to import topic_modeling. must be run from final_project/ dir
lib_path = os.path.abspath(os.path.join('.'))
sys.path.append(lib_path)

import numpy as np
import pylab
import pickle
from collections import Counter
from topic_modeling.topic_model import preprocess_docs
from pca_plot import reduce_dim


def doc_to_noun_bow(doc_nouns, top_nouns):
    top_nouns_set = set(top_nouns)
    matching_words = filter(lambda word: word in top_nouns_set, doc_nouns)
    counts = Counter(matching_words)

    return np.array([counts[noun]/float(len(doc_nouns) + 1) for noun in top_nouns])


def get_most_freq_words(docs, n=100):
    all_words = reduce(lambda x, y: x + y, docs)
    word_counts = Counter(all_words)
    return [word for word, count in word_counts.most_common(n)]


def get_and_save_word_counts(docs, vocab, outfile="metrics/word_counts.csv"):
    word_counts_by_doc = np.vstack([doc_to_noun_bow(doc, vocab) for doc in docs])
    np.savetxt(outfile, word_counts_by_doc, delimiter=",")
    return word_counts_by_doc


def plot_pca_noun_data(noun_counts_by_doc, labels, outfile):
    colors = ["green" if label else "red" for label in labels]

    reduced_data = reduce_dim(noun_counts_by_doc, 2)
    pylab.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors)
    # pylab.ylim(-10, 10)
    # pylab.xlim(-10, 10)
    pylab.ylabel("Count Data Principal Component 2")
    pylab.xlabel("Count Data Principal Component 1")
    pylab.title("Word Count Data Plotted By PCA: Nonromantic Lexicon Words")
    pylab.savefig(outfile)
    pylab.show()


if __name__ == '__main__':
    label_file = sys.argv[3]
    png_out_file = sys.argv[4]

    if sys.argv[1] == "-pos":
        postags_file = sys.argv[2]
        doc_nouns = preprocess_docs(doc_texts=None, postags_file=postags_file)
        noun_counts = get_and_save_word_counts(doc_nouns, get_most_freq_words(doc_nouns, 100))
    elif sys.argv[1] == "-csv":
        csv_file = sys.argv[2]
        noun_counts = np.loadtxt(csv_file, dtype=int, delimiter=",")

    labels = np.loadtxt(label_file, dtype=int, delimiter=",")
    labels = labels[:np.shape(noun_counts)[0]]

    plot_pca_noun_data(noun_counts, labels, png_out_file)