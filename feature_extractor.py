#!/usr/bin/python

import math
import pickle
import re
import sys
from collections import Counter

import numpy as np
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


# from topic_modeling.topic_model import get_top_words_in_topics, stem_all_words

###########################################################################
class Relationship:
    people_regex = "((\d+(\s+|,|\?|/)?(\s?|s)([MmFf]{1}([^/a-z])|([Mm]/[Ff])))|((\s|\[|\()[MmFf](\s|/)?\d\d)((\s|\]|\)))|((\[|\()\d\d)(\]|\)))"
    duration_regex = "((of|duration|over)?\s+(\d|one|a)+\s*\S*\s*(Year|years|year|yr|yrs|months|month|weeks))"

    def __init__(self, titletext):
        self.people = self.extract_people(titletext)
        self.duration = self.extract_duration(titletext)

    # Number, gender of people can be obtained from title
    def extract_people(self, titletext):
        res = re.findall(self.people_regex, titletext)
        people = []
        for p in res:
            if not (p is None):
                people.append(Person(p[0]))
        return people

    # Duration of relationship obtained from title
    # 1. Raw duration in months of relationship
    # 2. Boolean, < or > 6 months
    def extract_duration(self, titletext):
        res = re.search(self.duration_regex, titletext)
        if res is None:
            return None
        else:
            # print res.group(3)
            return Duration(res.group(3))

    # 1. Number of people
    def extract_num_people(self):
        return len(self.people)

    # 2. Number of each gender
    def extract_num_fem(self):
        filtered = [1 for p in self.people if p.gender == 'F']
        return sum(filtered)

    def extract_num_male(self):
        filtered = [1 for p in self.people if p.gender == 'M']
        return sum(filtered)

    # 3. Hetero-normative
    # Boolean: detect if only M & F in list of people
    def extract_hetero(self):
        if len(self.people) != 2:
            return False
        person1, person2 = self.people
        return (person1.gender == 'F' and person2.gender == 'M') or \
               (person1.gender == 'M' and person2.gender == 'F')

    def age_difference(self):
        if len(self.people) != 2:
            return 100
        else:
            person1, person2 = self.people
            return abs(person1.age - person2.age)

    # 5?. Boolean occurrence of: girlfriend, boyfriend, wife,
    # husband, mother, father, sister, brother, uncle, aunt

    def get_all_features(self):
        features = list()
        features.append(self.extract_num_people())
        features.append(self.extract_num_fem())
        features.append(self.extract_num_male())
        features.append(int(self.extract_hetero()))
        features.append(self.age_difference())

        # features += self.duration.get_all_features()
        # for p in self.people:
        #     features += p.get_all_features()
        return features


###########################################################################
class Person:
    def __init__(self, text):
        self.gender = None
        self.age = None
        self.preprocess(text)

        if "m" in text.lower():
            self.gender = "M"
        else:
            self.gender = "F"

        age_match = re.search(r"\d+", text)
        if age_match is None:
            raise Exception("Could not parse age for person.")

        self.age = int(age_match.group(0))
        # print self.age

        assert self.gender == 'F' or self.gender == 'M'
        # assert self.age > 0 and self.age < 100

    def preprocess(self, text):
        pass

    def get_all_features(self):
        features = [self.gender, self.age]
        return features


###########################################################################
class Duration:
    """
    Measured in months
    """

    def __init__(self, text):
        self.length = self.preprocess(text)
        self.text = text

    def preprocess(self, text):
        return 0
        # todo

    # 1. boolean if over a year
    def over_a_year(self):
        return self.length >= 12

    # 2. boolean if over a month
    def over_a_month(self):
        return self.length > 1

    # 3. exact length in months
    def extract_length(self):
        return self.length

    def get_all_features(self):
        features = list()
        features.append(self.over_a_year())
        features.append(self.over_a_month())
        features.append(self.extract_length())

        return features


###########################################################################
class Post:
    edit_regex = "\b(EDIT|edit)\b\n"
    tldr_regex = "(\*\*)?\b(tl;dr|TLDR|TL;DR)\b(\*\*)?"

    def __init__(self, row):
        self.text = row['selftext']
        self.gilded = row['gilded']
        self.upvotes = row['score']
        self.comments = row['num_comments']

    # if gilded
    def extract_gilded(self):
        return self.gilded

        # num upvotes
    def extract_upvotes(self):
        return self.upvotes

        # num comments
    def extract_comments(self):
        return self.comments

    # Update: or Edit: in body of post
    def extract_edit(self):
        res = re.search(self.edit_regex, self.text)
        return res

    # Tldr words
    def extract_tldr(self):
        res = re.search(self.tldr_regex, self.text)
        return res

    def get_all_features(self):
        features = [self.extract_gilded(),
                    self.extract_upvotes(),
                    self.extract_comments()]
        return features

class FeatureExtractor:
    def __init__(self, data, topic_words_file="topic_modeling/topic_top_words.txt"):
        self.feature_matrix = None
        self.data = data
        self.topic_words_file = topic_words_file

        with open("wordnet-lexicon/romantic_words", 'rb') as f:
            romantic_words = [w.strip('\n') for w in f.readlines()]
        with open("wordnet-lexicon/nonromantic_words", 'rb') as f:
            nonromantic_words = [w.strip('\n') for w in f.readlines()]

        self.lexicon = romantic_words + nonromantic_words
        self.topics_top_words = []
        self.topics_top_words = get_top_words_in_topics(model_file=None, text_file=self.topic_words_file)


    def get_lexicon_features(self, post_text):
        if post_text == None:
            return np.array([])
        tokenized = word_tokenize(post_text)
        # bigrams = window(tokenized, 2)
        # bigrams = ['_'.join(b) for b in bigrams]
        # trigrams = window(tokenized, 3)
        # trigrams = ['_'.join(t) for t in trigrams]
        return self.get_normed_word_counts(self.lexicon, tokenized)

    # Features from topic modeling
    @staticmethod
    def get_normed_word_counts(top_topic_words, post_words):
        post_word_count = float(len(post_words))
        post_words = [p.lower() for p in post_words]

        if post_word_count == 0:
            sys.stderr.write("Warning: empty post text.\n")
            return np.zeros(len(top_topic_words))

        matching_words = filter(lambda word: word in top_topic_words, post_words)
        counts = Counter(matching_words)
        return np.array([float(counts[word])/post_word_count for word in top_topic_words])

    def get_topic_model_features(self, posttext):
        post_words = stem_all_words(word_tokenize(posttext))

        # sys.stderr.write("Post length: " + str(len(post_words)) + "\n")

        return self.get_normed_word_counts(self.topics_top_words, post_words)

    def extract_post_features(self, post):
        # print "Extracting post features"
        post_obj = Post(post)

        features = np.array([])
        post_text = post["selftext"]
        # features = np.concatenate((features, post_obj.get_all_features()))
        features = np.concatenate((features, self.get_topic_model_features(post_text)))
        features = np.concatenate((features, self.get_lexicon_features(post_text)))
        features = np.concatenate((features, self.get_lexicon_features(post_obj.extract_tldr())))
        features = np.concatenate((features, self.get_lexicon_features(post_obj.extract_edit())))  # weigh these more?
        return features

    def extract_title_features(self, titletext):
        # print "Extracting title features"
        relationship = Relationship(titletext)
        # print relationship
        features = np.array([])
        features = np.concatenate((features, relationship.get_all_features()))
        features = np.concatenate((features, self.get_lexicon_features(titletext)))
        return features

    def extract_all_features_from_row(self, datarow):
        # features = self.extract_post_features(datarow["selftext"])
        # return features
        return np.concatenate((self.extract_post_features(datarow), self.extract_title_features(datarow["title"])))

    def mean_normalize_all_features(self):
        if self.feature_matrix is None:
            return

        for col_label in self.feature_matrix:
            column = self.feature_matrix[col_label]
            if np.max(column) == 1 and np.min(column) == 0:
                continue
            else:
                avg = np.mean(column)
                stdev = math.sqrt(np.var(column))
                self.feature_matrix[col_label].apply(lambda x: (x - avg)/stdev)

    # Extracts all the features at once. This is the only function that should be used by the classifier,
    # unless testing the efficacy of differing feature sets.
    def extract_full_feature_matrix(self):
        self.features = np.vstack([self.extract_all_features_from_row(row) for i, row in self.data.iterrows()])
        self.mean_normalize_all_features()
        return self.features


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


def save_model_and_top_words(model, model_file, words_file, num_topics=2):
    with open(model_file, "w") as f:
        pickle.dump(model, f)

    write_top_words_to_file(topic_model, num_topics, words_file)


def stem_all_words(words):
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(word) for word in words]


def get_top_words_in_topics(model_file="topic_modeling/lda_train.model", text_file="topic_modeling/topic_top_words.txt",
                            ntopics=2, nwords=20):
    if not (model_file is None):
        topic_model = load_model_from_file(model_file)
        return [[word for word, prob in topic_model.show_topic(i, nwords)] for i in range(ntopics)]
    elif not (text_file is None):
        with open(text_file) as f:
            return [word.strip() for word in f.readlines()]
    else:
        raise Exception('No model file or text file provided for getting topic words.')
