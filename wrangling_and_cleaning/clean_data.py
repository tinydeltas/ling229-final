#!/usr/bin/python
import sys
import pandas as pan

romantic_tags = ["Romantic", "Relationships", "Breakups", "Dating", "Infidelity"]
nonromantic_tags = ["Non-Romantic", "Personal Issues"]

def romance_tag(tag):
    if tag in romantic_tags:
        return "Romantic"
    elif tag in nonromantic_tags:
        return "NonRomantic"
    else:
        return "Update"

if __name__ == '__main__':
    data = pan.read_csv(sys.argv[1])
    data = data.iloc[1:, :10]

    new_label = [romance_tag(x) for x in data["link_flair_text"]]
    data["is_romantic"] = pan.Series(new_label, index=data.index)

    #print data[data["is_romantic"] == "Update"]
    data = data.drop(data[data["is_romantic"] == "Update"].index)

    data.to_csv("data/cleaned_tagged_relationships_data.csv", index_label=False)