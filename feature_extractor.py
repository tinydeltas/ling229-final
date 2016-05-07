#!/usr/bin/python

import re

###########################################################################
class Relationship:
    people_regex = "((\d+(\s+|,|\?|/)?(\s?|s)([MmFf]{1}([^/a-z])|([Mm]/[Ff])))|((\s|\[|\()[MmFf](\s|/)?\d\d)((\s|\]|\)))|((\[|\()\d\d)(\]|\)))"
    duration_regex = "((of|duration|over)?\s+(\d|one|a)+\s*\S*\s*(Year|years|year|yr|yrs|months|month|weeks))"

    def __init__(self, titletext):
        self.people = self.extract_people(titletext)
        self.duration = self.extract_duration(titletext)

    # Number, gender of people can be obtained from title
    def extract_people(self, titletext):
        res = re.search(self.people_regex, titletext)
        people = []
        for p in res:
            people.append(Person(p))
        return people

    # Duration of relationship obtained from title
    # 1. Raw duration in months of relationship
    # 2. Boolean, < or > 6 months
    def extract_duration(self, titletext):
        res = re.search(self.duration_regex, titletext)
        return Duration(res)

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

    # 5?. Boolean occurrence of: girlfriend, boyfriend, wife,
    # husband, mother, father, sister, brother, uncle, aunt

    def get_all_features(self):
        features = list()
        features.append(self.extract_num_people())
        features.append(self.extract_num_fem())
        features.append(self.extract_num_male())
        features.append(self.extract_hetero())

        features += self.duration.get_all_features()
        for p in self.people:
            features += p.get_all_features()
        return features


###########################################################################
class Person:
    def __init__(self, text):
        self.gender = None
        self.age = None
        self.preprocess(text)
        assert self.gender == 'F' or self.gender == 'M'
        assert self.age > 0 and self.age < 100

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


def extract_post_features(post):
    print "Extracting post features"
    post = Post(post['selftext'])
    return post.get_all_features()


def extract_title_features(titletext):
    print "Extracting title features"
    relationship = Relationship(titletext)
    print relationship
    return relationship.get_all_features()
