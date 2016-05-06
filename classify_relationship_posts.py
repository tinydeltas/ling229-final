#!/usr/bin/python

import numpy as np
import pandas as pan
from classify_util import *
from feature_extractor import *

def get_options():
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train", default=None,
                      help="training data file", metavar="FILE")
    parser.add_option("-m", "--modelfile", dest="modelfile", default=None,
                        help="existing model file location")
    parser.add_option("-e", "--eval", dest="eval", default=None,
                       help="test data file", metavar="FILE")

    (options, args) = parser.parse_args()
    check_mandatory_options(parser, options, ["modelfile"])

    return options

def read_csv_data(csv_file, maxrows = None):
    if maxrows:
        return pan.read_csv(csv_file, nrows = maxrows, encoding='utf-8')
    return pan.read_csv(csv_file, encoding='utf-8')

def get_features_and_label_from_row(data_row):
    return extract_title_features(row["title"]) + extract_post_features(row["selftext"]) + [row["is_romantic"]]

def train_model(train_data, model_file):
    training_events = train_data.apply(lambda row : get_features_and_label_from_row(row), axis = 1)



if __name__ == '__main__':
    opts = get_options()

    if not opts.train and not opts.eval:
        raise Exception("Must supply either a training file or an evaluation file to the script.")

    # run training if training file given
    if opts.train:
        data = read_csv_data(opts.train)


    # output predictions on the test data if data was given
    if opts.eval:
        pass