#!/usr/bin/bash
python classify_relationship_posts.py -m classifier/post_classifier.model -t data/train.csv -e data/dev.csv
python score.py -g data/dev_labels.csv -p predictions/test_predictions.csv