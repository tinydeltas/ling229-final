#!/usr/bin/python

import numpy as np
import pandas as pan
from classify_util import *

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
        return pan.read_csv(csv_file, nrows = maxrows)
    return pan.read_csv(csv_file)

if __name__ == '__main__':
    opts = get_options()

    if not opts.train and not opts.eval:
        raise Exception("Must supply either a training file or an evaluation file to the script.")

    # run training if training file given
    if opts.train:
        data = read_csv_data(opts.train, 10)
        print data

    # output predictions on the test data if data was given
    if opts.eval:
        pass