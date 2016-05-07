#!/usr/bin/python

from classify_util import *


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
def extract_post_features(posttext):
    pass

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
