#!/usr/bin/python

#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#############################################################################

from __future__ import division
import sys
from optparse import OptionParser
from classify_util import *

#############################################################################
# Set up the options

parser = OptionParser()
parser.add_option("-g", "--gold", dest="gold",
                  help="use gold labels in FILE", metavar="FILE")
parser.add_option("-p", "--predict", dest="predict",
                  help="score predicted labels in FILE", metavar="FILE")

(options, args) = parser.parse_args()
check_mandatory_options(parser, options, ['gold'])

#############################################################################
# Use the options to set the input and output files appropriately

gold_file = file(options.gold)
prediction_file = sys.stdin
if options.predict != None:
    prediction_file = file(options.predict)

#############################################################################
# Do the scoring.

# Slurp in the labels from each file as lists
gold = [x.strip().split(',')[-1] for x in gold_file.readlines()]
predicted = [x.strip().split(' ')[0] for x in prediction_file.readlines()]

if len(gold) != len(predicted):
    print "ERROR: Different number of gold and predicted labels!"
    print "\tNum gold labels:", len(gold)
    print "\tNum predicted labels:", len(predicted)
    print "Exiting."
    sys.exit

# Zip the gold and predicted lists together and test for equality,
# then sum to get the number that matched.
num_correct = sum([x[0]==x[1] for x in zip(gold,predicted)])

accuracy = num_correct/len(gold) * 100.0
print "Accuracy: %.2f" % accuracy
