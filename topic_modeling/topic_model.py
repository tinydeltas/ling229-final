#!/usr/bin/python

import sys
import pickle
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from pos_tag_selftext import add_pos_tags_to_english_docs


def get_nouns(pos_tagged_doc):
    output = []
    for tok, tag in pos_tagged_doc:
        if tag == "NN":
            output.append(tok)
    return output


def stem_all_words(words):
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(word) for word in words]


def preprocess_docs(doc_texts, postags_file = None, save_postags = True, postags_file_name = "postags_obj"):
    pos_tagged_docs = []
    if postags_file:
        with open(postags_file) as f:
            pos_tagged_docs = pickle.load(f)
    else:
        if not doc_texts:
            raise Exception("No POS tag file or raw text passed to document processor.")

        pos_tagged_docs = add_pos_tags_to_english_docs(doc_texts)
        if save_postags:
            with open(postags_file_name, "w") as f:
                pickle.dump(pos_tagged_docs, f)

    return [stem_all_words(get_nouns(doc)) for doc in pos_tagged_docs]


def train_lda_model(postag_file, topics = 2):
    nouns_in_docs = preprocess_docs(doc_texts=None, postags_file=postag_file)

    corpora_dict = corpora.Dictionary(nouns_in_docs)

    corpora_dict.filter_extremes(no_below = 3, no_above = 0.65)

    corpus = [corpora_dict.doc2bow(doc) for doc in nouns_in_docs]

    topic_model = models.ldamodel.LdaModel(corpus, num_topics = topics, id2word = corpora_dict, passes = 20)

    return topic_model


def write_top_words_to_file(lda_model, num_topics, filename):
    with open(filename, "w") as f:
        for i in range(num_topics):
            for word, prob in lda_model.show_topic(i, 20):
                f.write(word + " ")
                f.write("\n")
        f.flush()


def load_model_from_file(filename):
    with open(filename) as f:
        return pickle.load(f)


def save_model_and_top_words(model, model_file, words_file, num_topics = 2):
    with open(model_file, "w") as f:
        pickle.dump(model, f)

    write_top_words_to_file(topic_model, num_topics, words_file)


def get_top_words_in_topics(model_file="topic_modeling/lda_train.model", text_file="topic_modeling/topic_top_words.txt", ntopics = 2, nwords = 20):
    if not (model_file is None):
        topic_model = load_model_from_file(model_file)
        return [[word for word, prob in topic_model.show_topic(i, nwords)] for i in range(ntopics)]
    elif not (text_file is None):
        with open(text_file) as f:
            return [word.strip() for word in f.readlines()]
    else:
        raise Exception('No model file or text file provided for getting topic words.')


if __name__ == '__main__':
    pos_tagged_docs_file = sys.argv[1]

    model_output_file = "topic_modeling/lda_train.model"
    top_words_output_file = "topic_modeling/topics.txt"

    if len(sys.argv) > 2 and sys.argv[2] == "-o":
        model = load_model_from_file(model_output_file)
        write_top_words_to_file(model, 2, sys.argv[3])
        sys.exit(0)

    print "Processing data and training model."

    topic_model = train_lda_model(pos_tagged_docs_file, 2)

    print "Saving model as " + model_output_file

    save_model_and_top_words(topic_model, model_output_file, top_words_output_file)
