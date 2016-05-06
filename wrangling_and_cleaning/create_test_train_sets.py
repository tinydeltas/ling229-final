#!/usr/bin/python
import sys
import pandas as pan

if __name__ == '__main__':
    data = pan.read_csv(sys.argv[1])

    data_to_use = data.sample(n = 1200)

    train_data = data_to_use.iloc[:1000, :]
    test_data = data_to_use.iloc[1000:1100, :]
    dev_data = data_to_use.iloc[1100:, :]

    train_data.to_csv("data/train.csv", index_label = False)
    test_data.to_csv("data/test.csv", index_label = False)
    dev_data.to_csv("data/dev.csv", index_label = False)