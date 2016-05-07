#!/usr/bin/python

import sys
import pickle
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models


def get_nouns(pos_tagged_doc):
    output = []
    for tok, tag in pos_tagged_doc:
        if tag == "NN":
            output.append(tok)
    return output


def stem_all_words(words):
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(word) for word in words]


def train_lda_model(pos_tagged_docs, topics = 2):
    nouns_in_docs = [stem_all_words(get_nouns(doc)) for doc in pos_tagged_docs]

    corpora_dict = corpora.Dictionary(nouns_in_docs)

    corpora_dict.filter_extremes(no_below = 3, no_above = 0.65)

    corpus = [corpora_dict.doc2bow(doc) for doc in nouns_in_docs]

    topic_model = models.ldamodel.LdaModel(corpus, num_topics = topics, id2word = corpora_dict, passes = 20)

    return topic_model


def write_top_words_to_file(lda_model, num_topics, filename):
    with open(filename, "w") as f:
        for i in range(num_topics):
            f.write("Topic " + str(i) + "\n")
            for word, prob in lda_model.show_topic(i, 20):
                print(word)
                f.write(word + " ")
                f.write(str(prob))
                f.write("\n")
            f.write("\n")
        f.flush()


def load_model_from_file(filename):
    with open(filename) as f:
        return pickle.load(f)


if __name__ == '__main__':
    pos_tagged_docs_file = sys.argv[1]

    model_output_file = "topic_modeling/lda_train.model"
    top_words_output_file = "topic_modeling/topics.txt"

    if len(sys.argv) > 2 and sys.argv[2] == "-o":
        model = load_model_from_file(model_output_file)
        write_top_words_to_file(model, 2, sys.argv[3])
        sys.exit(0)

    print "Loading the POS tags."

    with open(pos_tagged_docs_file) as f:
        pos_tagged_docs = pickle.load(f)

    print "Cleaning data and training model."

    topic_model = train_lda_model(pos_tagged_docs, 2)

    print "Saving model as " + model_output_file

    with open(model_output_file, "w") as f:
        pickle.dump(topic_model, f)

    write_top_words_to_file(topic_model, 2, top_words_output_file)