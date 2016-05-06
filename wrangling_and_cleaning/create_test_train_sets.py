#!/usr/bin/python
import sys
import pandas as pan

if __name__ == '__main__':
    data = pan.read_csv(sys.argv[1])

    data_to_use = data.sample(10200)

    train_data = data_to_use.iloc[:10000, :]
    test_data = data_to_use.iloc[10000:10100, :]
    dev_data = data_to_use.iloc[10100:, :]

    train_data.to_csv("data/train.csv")
    test_data.to_csv("data/test.csv")
    dev_data.to_csv("data/dev.csv")