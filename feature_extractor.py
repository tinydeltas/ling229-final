#!/usr/bin/python

import re
import sys
import math
import numpy as np
import pandas as pan
from nltk import word_tokenize
from collections import Counter
from topic_modeling.topic_model import get_top_words_in_topics, stem_all_words, load_model_from_file


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
        features = [self.extract_upvotes(),
                    self.extract_comments()]
        return features


class FeatureExtractor:
    def __init__(self, data, topic_words_file="topic_modeling/topic_top_words.txt",
                 topic_model_file="topic_modeling/lda_train.model"):
        self.feature_matrix = None
        self.data = data
        self.topic_words_file = topic_words_file
        self.topic_model = load_model_from_file(topic_model_file)

        with open("wordnet-lexicon/romantic_words", 'rb') as f:
            romantic_words = [w.strip() for w in f.readlines()]
        with open("wordnet-lexicon/nonromantic_words", 'rb') as f:
            nonromantic_words = [w.strip() for w in f.readlines()]

        self.lexicon = romantic_words + nonromantic_words
        self.romantic_lexicon = romantic_words
        self.nonromantic_lexicon = nonromantic_words
        self.topics_top_words = []
        self.topics_top_words = get_top_words_in_topics(model_file=None, text_file=self.topic_words_file)

    def get_lexicon_features(self, post_tokens):
        if post_tokens is None or len(post_tokens) == 0:
            sys.stderr.write("Warning: No post text passed to get lexicon features.\n")
            return np.zeros(4 + len(self.lexicon))

        romantic_wc = self.get_word_counts(self.romantic_lexicon, post_tokens)
        nonromantic_wc = self.get_word_counts(self.nonromantic_lexicon, post_tokens)

        romantic_sum = np.sum(romantic_wc) + 1.0
        nonromantic_sum = np.sum(nonromantic_wc) + 1.0

        post_len = float(len(post_tokens))

        romantic_lexicon_ratio = romantic_sum / (romantic_sum + nonromantic_sum)
        nonromantic_lexicon_ratio = nonromantic_sum / (romantic_sum + nonromantic_sum)

        summary_features = np.array([romantic_lexicon_ratio, nonromantic_lexicon_ratio, romantic_sum/post_len, nonromantic_sum/post_len])
        lexicon_features = np.concatenate((romantic_wc, nonromantic_wc))
        normalizing_func = np.vectorize(lambda x: x/post_len, otypes=[np.float])
        lexicon_features = normalizing_func(lexicon_features)

        return np.concatenate((summary_features, lexicon_features))

    # Features from topic modeling
    @staticmethod
    def get_word_counts(top_topic_words, post_words, normed=True):
        post_word_count = float(len(post_words))
        post_words = [p.lower() for p in post_words]

        if post_word_count == 0:
            sys.stderr.write("Warning: empty post text.\n")
            return np.zeros(len(top_topic_words))

        top_topic_words_set = set(top_topic_words)

        matching_words = filter(lambda word: word in top_topic_words_set, post_words)
        counts = Counter(matching_words)

        if normed:
            norm_factor = post_word_count
        else:
            norm_factor = 1.0

        return np.array([float(counts[word])/norm_factor for word in top_topic_words])

    def get_topic_model_features(self, post_tokens):
        post_stems = stem_all_words(post_tokens)

        # sys.stderr.write("Post length: " + str(len(post_words)) + "\n")
        # print self.topic_model.id2word

        post_bow = self.topic_model.id2word.doc2bow(post_stems)
        topic_distrs = self.topic_model.get_document_topics(post_bow)
        topic_distr_feats = np.array([0.0, 0.0])
        for topic_idx, prob in topic_distrs:
            topic_distr_feats[topic_idx] = prob

        return np.concatenate((self.get_word_counts(self.topics_top_words, post_stems), topic_distr_feats))

    def extract_post_features(self, post):
        # print "Extracting post features"
        post_obj = Post(post)

        features = np.array([])
        post_text = post["selftext"]

        post_tokens = word_tokenize(post_text)

        features = np.concatenate((features, self.get_topic_model_features(post_tokens),
                                   self.get_lexicon_features(post_tokens),
                                   np.array(post_obj.get_all_features())))

        # features = np.concatenate((features, self.get_lexicon_features(post_obj.extract_tldr())))
        # features = np.concatenate((features, self.get_lexicon_features(post_obj.extract_edit())))  # weigh these more?
        return features

    def extract_title_features(self, titletext):
        # print "Extracting title features"
        relationship = Relationship(titletext)
        # print relationship
        features = np.concatenate((np.array(relationship.get_all_features()),
                                   self.get_lexicon_features(titletext)))
        return features

    def extract_all_features_from_row(self, datarow):
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
        self.feature_matrix = np.vstack([self.extract_all_features_from_row(row) for i, row in self.data.iterrows()])
        # self.mean_normalize_all_features()
        return self.feature_matrix


# When run as a script, extract and save features
if __name__ == '__main__':
    data_file = sys.argv[1]
    out_file = sys.argv[2]

    data = pan.read_csv(data_file, encoding='utf-8')
    extractor = FeatureExtractor(data)
    feat_matrix = extractor.extract_full_feature_matrix()
    np.savetxt(out_file, feat_matrix, delimiter=",")