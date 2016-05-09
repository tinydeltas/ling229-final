from nltk.corpus import wordnet as wn

romantic_words = {'relationship', 'romantic, ''love', 'sex', 'cheating', 'dating', 'crush',
                  'boyfriend', 'girlfriend', 'husband', 'wife', 'monogamous', 'serious',
                  'fiance', 'fiancee', 'partner', 'ex', 'bf', 'gf', 'snoop',
                  'dumped', 'married', 'wedding', 'infidelity', 'breakup', 'broke_up',
                  'date', 'our', 'attraction', 'attracted', 'over_her', 'over_him'}

nonromantic_words = {'boss', 'manager', 'co-worker', 'colleague', 'teacher',
                     'work', 'job', 'jobs', 'applied', 'company', 'school',
                     'friend', 'cousin', 'aunt', 'uncle', 'college', 'property',
                     'niece', 'nephew', 'son', 'daughter', 'kid', 'child',
                     'parents', 'mom', 'mother', 'dad', 'father', 'grandparents',
                     'neighbor', 'gift', 'roommate', 'roommates', 'acquaintance',
                     'best_friend', 'close_friend', 'best_mate', 'bromance',
                     'close', 'neighborhood', 'platonic', 'friendly', 'dysfunctional'}


def find_xnonyms(set_list):
    expansion = set({})
    for word in set_list:
        syns = wn.synsets(word)
        for syn in syns:
            for l in syn.lemmas():
                expansion.add(l.name().lower())
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
