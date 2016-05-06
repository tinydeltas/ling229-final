import sys
import pandas as pan
from nltk import pos_tag, word_tokenize
from classify_relationship_posts import *


def add_pos_tags_to_english_docs(docs):
    output = []
    for doc in docs:
        print("*")
        tagged_toks = pos_tag(word_tokenize(doc))
        # recreate sentence with words attached to their POS
        #output.append(" ".join(map(lambda tag_tuple: "-".join(tag_tuple), tagged_toks)) + "\n")
        output.append(tagged_toks)
    return output

if __name__ == '__main__':
    n = int(sys.argv[1])
    english_filename = sys.argv[2]

    english_docs = list(read_csv_data(english_filename, maxrows = n)["selftext"])

    print(english_docs[0].encode('utf-8'))

    english_docs = [english_docs[i].encode('utf-8') for i in range(len(english_docs))]

    #english_sents_tagged = add_pos_tags_to_english_docs(english_docs)

    #print(english_sents_tagged)
    #with open("data/english_tagged" + str(n), "w") as outfile:
    #    outfile.writelines(english_sents_tagged)