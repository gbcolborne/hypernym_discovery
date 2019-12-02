import argparse
import numpy as np

""" Analyze length of terms in a list. """

parser = argparse.ArgumentParser()
parser.add_argument("terms", help="Path of list of terms (one per line)")
args = parser.parse_args()

with open(args.terms) as f:
    lens = []
    for line in f:
        term = line.strip()
        if term:
            lens.append(len(term.split(" ")))
print("Nb terms read: {}")
print("Minimum length: {} words".format(min(lens)))
print("Maximum length: {} words".format(max(lens)))
print("Mean length: {} words".format(np.mean(lens)))
print("Median length: {} words".format(np.median(lens)))
