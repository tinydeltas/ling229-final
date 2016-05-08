#!/usr/bin/python

import sys
import pickle
import pandas as pan
from nltk import pos_tag, word_tokenize


def add_pos_tags_to_english_docs(docs):
    output = []
    for doc in docs:
        if not (type(doc) is str):
            output.append([])
            continue

        sys.stdout.write("*")
        tagged_toks = pos_tag(word_tokenize(doc.decode('utf-8')))
        output.append(tagged_toks)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return output

if __name__ == '__main__':
    n = int(sys.argv[1])
    english_filename = sys.argv[2]
    outfile_name = sys.argv[3]
    
    print("Reading data file.")

    english_docs = list(pan.read_csv(english_filename, nrows=n, encoding='utf-8')["selftext"])

    english_docs = [english_docs[i].encode('utf-8') for i in range(len(english_docs))]

    print("Starting tagging.")

    english_sents_tagged = add_pos_tags_to_english_docs(english_docs)

    with open(outfile_name + str(n), "w") as outfile:
        pickle.dump(english_sents_tagged, outfile)