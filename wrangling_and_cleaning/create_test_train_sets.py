#!/usr/bin/python
import sys
import pandas as pan

if __name__ == '__main__':
    data = pan.read_csv(sys.argv[1], nrows = 1200)

    data_to_use = data.sample(n = 1200)

    train_data = data_to_use.iloc[:1000, 1:]
    test_data = data_to_use.iloc[1000:1100, 1:]
    dev_data = data_to_use.iloc[1100:, 1:]

    print(test_data)

    train_data.to_csv("data/train.csv")
    test_data.to_csv("data/test.csv")
    dev_data.to_csv("data/dev.csv")