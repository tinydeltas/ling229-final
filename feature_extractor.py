#!/usr/bin/python

import numpy as np
from nltk import word_tokenize
from topic_modeling.topic_model import get_top_words_in_topics, stem_all_words
from collections import Counter


###########################################################################
def extract_title_features(titletext):
    pass

# 1. (Absolute) age difference between people in post
def extract_ages(titletext):
    pass

# Number, gender of people in title
# 1. Number of people
# 2. Number of each gender
def extract_people(titletext):
    regex ="((\d+(\s+|,|\?|/)?(\s?|s)([MmFf]{1}([^/a-z])|([Mm]/[Ff])))|((\s|\[|\()[MmFf](\s|/)?\d\d)((\s|\]|\)))|((\[|\()\d\d)(\]|\)))"

# Duration of relationship in title
# 1. Raw duration in months of relationship
# 2. Boolean, < or > 6 months
def extract_duration(titletext):
    duration_regex="((of|duration|over)?\s+(\d|one|a)+\s*\S*\s*(Year|years|year|yr|yrs|months|month|weeks))"

# 1. Boolean occurrence of: girlfriend, boyfriend, wife, husband, mother, father, sister, brother, uncle, aunt
def extract_people(titletext):
    pass

# Heteronormativity
# 1. Boolean: detect if only M & F in list of people
def extract_type(titletext):
    pass

###########################################################################


# Features from topic modeling
def get_normed_word_counts(top_topic_words, post_words):
    post_word_count = float(len(post_words))
    matching_words = filter(lambda word: word in top_topic_words, post_words)
    counts = Counter(matching_words)
    return np.array([float(counts[word])/post_word_count for word in top_topic_words])


def get_topic_model_features(posttext):
    top_topic_words = []
    for words in get_top_words_in_topics():
        top_topic_words += words

    post_words = stem_all_words(word_tokenize(posttext))

    return get_normed_word_counts(top_topic_words, post_words)



# Update: or Edit: in body of post
def extract_edit(posttext):
    edit_regex = "\b(EDIT|edit)\b\n"
    pass

# Tldr words
def extract_tldr(posttest):
    tldr_regex = "(\*\*)?\b(tl;dr|TLDR|TL;DR)\b(\*\*)?";

# Time created
# If gilded
# Upvotes
# Number of comments

def extract_post_features(posttext):
    features = get_topic_model_features(posttext)
    return features
