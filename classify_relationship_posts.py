#!/usr/bin/python

"""
MAIN DRIVER CODE FOR CLASSIFICATION
This class does the main classification and prediction task.
Usage:
classify_relationships_posts.csv -m [modelfile] -t [training file csv] -e [test file csv] -o [prediction output file]
Advanced options:
-p [probability prediction output file] -c [classifer type ("tree" or "logit")]
"""

import numpy as np
import pandas as pan
import pickle
from classify_util import *
from feature_extractor import FeatureExtractor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def get_options():
    """
    Gets the options for the script/
    :return: OptionParser.parse_args() options return value (dict of opts)
    """
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train", default=None,
                      help="training data file", metavar="FILE")
    parser.add_option("-m", "--modelfile", dest="modelfile", default=None,
                      help="existing model file location")
    parser.add_option("-e", "--eval", dest="eval", default=None,
                      help="test data file", metavar="FILE")
    parser.add_option("-o", "--outfile", dest="outfile", default="predictions/dev_predictions.csv",
                      help="prediction output file", metavar="FILE")
    parser.add_option("-p", "--probfile", dest="probfile", default="predictions/prediction_probs.csv",
                      help="prediction output file (class probability)", metavar="FILE")
    parser.add_option("-c", "--classifiertype", dest="classifiertype", default="tree",
                      help="prediction output file (class probability)", metavar="FILE")

    (options, args) = parser.parse_args()
    check_mandatory_options(parser, options, ["modelfile"])

    return options


class RelationshipPostClassifier:
    """
    Main class for classification and prediction
    """

    def __init__(self, classifier_type="tree"):
        self.data = None
        self.classifier_type = classifier_type
        self.classifier = None
        self.featureExtractor = None
        self.predictions = None

    def read_csv_data(self, csv_file, maxrows=None):
        """
        Read in data from given csv_file into self.data (pandas dataframe)
        maxrows limits number of read rows.
        :param csv_file:
        :param maxrows:
        :return: None
        """
        sys.stderr.write("Reading in data from " + csv_file + "\n")

        if maxrows:
            self.data = pan.read_csv(csv_file, nrows=maxrows, encoding='utf-8')
        else:
            self.data = pan.read_csv(csv_file, encoding='utf-8')

    def train_model(self, model_out_file):
        """
        Extract the features from self.data and train the classifier. Output pickled model to model_out_file
        :param model_out_file:
        :return: None
        """
        if self.data is None:
            raise Exception("Trying to train model without any data.")

        sys.stderr.write("Extracting features from data.\n")

        self.featureExtractor = FeatureExtractor(self.data)
        feature_matrix = self.featureExtractor.extract_full_feature_matrix()

        labels = np.array([0 if lab == "Romantic" else 1 for lab in self.data["is_romantic"]])

        sys.stderr.write("Training classifier.\n")

        self.classifier = LogisticRegression() if self.classifier_type == "logit" else DecisionTreeClassifier()
        self.classifier.fit(feature_matrix, labels)

        sys.stderr.write("Saving classifier.\n")

        with open(model_out_file, "w") as f:
            pickle.dump(self.classifier, f)

    def predict_model(self, model_file=None, output_file=None, output_probability_file=None):
        """
        Predict classes on self.data and output to output_file
        :param model_file: Model file to read model in from. Otherwise looks for self.classifier
        :param output_file: File to save predictions in
        :param output_probability_file: File to save predicted probabilities in
        :return: predicted classes (array)
        """
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

        if output_file is not None:
            np.savetxt(output_file, self.predictions, delimiter=",", fmt="%d")

        if output_probability_file is not None:
            pred_probs = self.classifier.predict_proba(feature_matrix)
            np.savetxt(output_probability_file, pred_probs, delimiter=",", fmt="%.3f")

        return self.predictions


# Main driver code
if __name__ == '__main__':
    opts = get_options()

    if not opts.train and not opts.eval:
        raise Exception("Must supply either a training file or an evaluation file to the script.")

    postClassifier = RelationshipPostClassifier(classifier_type=opts.classifiertype)

    # run training if training file given
    if opts.train:
        postClassifier.read_csv_data(opts.train)
        postClassifier.train_model(model_out_file=opts.modelfile)

    # output predictions on the test data if data was given
    if opts.eval:
        postClassifier.read_csv_data(opts.eval)
        if opts.train:
            postClassifier.predict_model(output_file=opts.outfile, output_probability_file=opts.probfile)
        else:
            postClassifier.predict_model(model_file=opts.modelfile, output_file=opts.outfile,
                                         output_probability_file=opts.probfile)
