#!/usr/bin/python

import numpy as np
import re
import sys
import math
from nltk import word_tokenize
from topic_modeling.topic_model import get_top_words_in_topics, stem_all_words
from collections import Counter


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
        print features
        return features


###########################################################################
class Post:
    edit_regex = "\b(EDIT|edit)\b\n"
    tldr_regex = "(\*\*)?\b(tl;dr|TLDR|TL;DR)\b(\*\*)?"

    def __init__(self, posttext):
        self.text = posttext

    # Update: or Edit: in body of post
    def extract_edit(self):
        res = re.search(self.edit_regex, self.text)
        return res

    # Tldr words
    def extract_tldr(self):
        res = re.search(self.tldr_regex, self.test)
        return res

    # if gilded
    def extract_gilded(self):
        pass

        # num upvotes

    def extract_upvotes(self):
        pass

        # num comments

    def extract_comments(self):
        pass

    def get_all_features(self):
        features = [self.extract_tldr(),
                    self.extract_edit(),
                    self.extract_gilded(),
                    self.extract_upvotes(),
                    self.extract_comments()]
        return features

# Time created
# If gilded
# Upvotes
# Number of comments


class FeatureExtractor:
    def __init__(self, data, topic_model_file="topic_modeling/lda_train.model"):
        self.feature_matrix = None
        self.data = data
        self.topic_model_file = topic_model_file

        self.topics_top_words = []
        topics_top_words = get_top_words_in_topics(model_file=self.topic_model_file)
        for words in topics_top_words:
            self.topics_top_words += words

    # Features from topic modeling
    @staticmethod
    def get_normed_word_counts(top_topic_words, post_words):
        post_word_count = float(len(post_words))

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

    def extract_post_features(self, post_text):
        # print "Extracting post features"
        post_obj = Post(post_text)

        features = np.array([])
        # features = np.concatenate((features, post_obj.get_all_features()))
        features = np.concatenate((features, self.get_topic_model_features(post_text)))

        return features

    def extract_title_features(self, titletext):
        # print "Extracting title features"
        relationship = Relationship(titletext)
        # print relationship
        return np.array(relationship.get_all_features())

    def extract_all_features_from_row(self, datarow):
        # features = self.extract_post_features(datarow["selftext"])
        # return features
        return np.concatenate((self.extract_post_features(datarow["selftext"]), self.extract_title_features(datarow["title"])))

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
