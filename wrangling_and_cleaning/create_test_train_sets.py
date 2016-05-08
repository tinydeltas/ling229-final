#!/usr/bin/python
import sys
import pandas as pan

if __name__ == '__main__':
    data = pan.read_csv(sys.argv[1], dtype={"selftext": unicode})

    data = data.dropna(axis=0)
    data = data[data.link_flair_text != "Relationships"]

    train_data = data.iloc[:10000, :]

    print data.shape

    data = data.iloc[10000:, :]

    data_label1 = data[data["is_romantic"] == "Romantic"]
    data_label2 = data[data["is_romantic"] == "NonRomantic"]

    print data_label1.shape
    print data_label2.shape

    test_data = data_label1.iloc[:500, :]
    dev_data = data_label1.iloc[500:1000, :]

    test_data = pan.concat((test_data, data_label2.iloc[:500, :]))
    dev_data = pan.concat((dev_data, data_label2.iloc[500:1000, :]))

    train_data.to_csv("data/train.csv")
    test_data.to_csv("data/test.csv")
    dev_data.to_csv("data/dev.csv")