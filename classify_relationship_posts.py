#!/usr/bin/python

import numpy as np
import pandas as pan
import pickle
from classify_util import *
from feature_extractor import FeatureExtractor
from sklearn.tree import DecisionTreeClassifier


def get_options():
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train", default=None,
                      help="training data file", metavar="FILE")
    parser.add_option("-m", "--modelfile", dest="modelfile", default=None,
                      help="existing model file location")
    parser.add_option("-e", "--eval", dest="eval", default=None,
                      help="test data file", metavar="FILE")
    parser.add_option("-o", "--outfile", dest="outfile", default="predictions/test_predictions.csv",
                      help="prediction output file", metavar="FILE")

    (options, args) = parser.parse_args()
    check_mandatory_options(parser, options, ["modelfile"])

    return options


class RelationshipPostClassifier:
    def __init__(self):
        self.data = None
        self.classifier = None
        self.featureExtractor = None
        self.predictions = None

    def read_csv_data(self, csv_file, maxrows=None):

        sys.stderr.write("Reading in data from " + csv_file + "\n")

        if maxrows:
            self.data = pan.read_csv(csv_file, nrows=maxrows, encoding='utf-8')
        else:
            self.data = pan.read_csv(csv_file, encoding='utf-8')

    def train_model(self, model_out_file):
        if self.data is None:
            raise Exception("Trying to train model without any data.")

        sys.stderr.write("Extracting features from data.\n")

        self.featureExtractor = FeatureExtractor(self.data)
        feature_matrix = self.featureExtractor.extract_full_feature_matrix()

        labels = np.array([0 if lab == "Romantic" else 1 for lab in self.data["is_romantic"]])

        sys.stderr.write("Training classifier.\n")

        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(feature_matrix, labels)

        sys.stderr.write("Saving classifier.\n")

        with open(model_out_file, "w") as f:
            pickle.dump(self.classifier, f)

    def predict_model(self, model_file = None, output_file = None):
        if not self.classifier:
            if not model_file:
                raise Exception("No model to predict with.")
            else:
                with open(model_file) as f:
                    self.classifier = pickle.load(f)

        if self.data is None:
            raise Exception("Trying to predict using model with no data loaded.")

        self.featureExtractor = FeatureExtractor(self.data)
        feature_matrix = self.featureExtractor.extract_full_feature_matrix()

        self.predictions = self.classifier.predict(feature_matrix)

        if output_file:
            np.savetxt(output_file, self.predictions, delimiter=",", fmt="%d")

        return self.predictions


# Main driver code
if __name__ == '__main__':
    opts = get_options()

    if not opts.train and not opts.eval:
        raise Exception("Must supply either a training file or an evaluation file to the script.")

    postClassifier = RelationshipPostClassifier()

    # run training if training file given
    if opts.train:
        postClassifier.read_csv_data(opts.train)
        postClassifier.train_model(model_out_file=opts.modelfile)

    # output predictions on the test data if data was given
    if opts.eval:
        postClassifier.read_csv_data(opts.eval)
        if opts.train:
            postClassifier.predict_model(output_file=opts.outfile)
        else:
            postClassifier.predict_model(model_file=opts.modelfile, output_file=opts.outfile)
