import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("path_queries", help="path of TSV file containing queries")
parser.add_argument("path_output")
args = parser.parse_args()

queries, qtypes = utils.load_queries(args.path_queries, normalize=False)
with open(args.path_output, 'w') as f:
    for q in queries:
        f.write(q+"\n")

