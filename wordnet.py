from nltk.corpus import wordnet as wn

romantic_words = {'relationship', 'love', 'sex', 'cheating', 'dating', 'crush',
                  'boyfriend', 'girlfriend', 'husband', 'wife', 'SO',
                  'fiance', 'fiancee', 'partner', 'ex', 'bf', 'gf'
                                                              'dumped', 'married', 'wedding', 'infidelity', 'breakup',
                  'date', 'our'}

nonromantic_words = {'boss', 'manager', 'co-worker', 'colleague', 'teacher',
                     'work', 'job', 'jobs', 'applied', 'company',
                     'friend', 'cousin', 'aunt', 'uncle',
                     'niece', 'nephew', 'son', 'daughter', 'kid', 'child',
                     'parents', 'mom', 'mother', 'dad', 'father', 'grandparents',
                     'neighbors', 'gift'}


def find_xnonyms(set_list):
    expansion = set({})
    for word in set_list:
        syns = wn.synsets(word)
        for syn in syns:
            for l in syn.lemmas():
                expansion.add(l.name())
    return expansion


romantic_words |= find_xnonyms(romantic_words)
print 'romantic: ', romantic_words
nonromantic_words |= find_xnonyms(nonromantic_words)
print 'nonromantic: ', nonromantic_words

with open('wordnet-lexicon/romantic_words', 'wb') as f:
    # pickle.dump(romantic_words, f)
    for r in romantic_words:
        f.write(r)
        f.write("\n")

with open('wordnet-lexicon/nonromantic_words', 'wb') as f:
    # pickle.dump(nonromantic_words, f)
    for r in nonromantic_words:
        f.write(r)
        f.write("\n")
