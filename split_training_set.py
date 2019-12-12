import argparse, random, os
import utils

""" Split training set into a random train/dev split. """

parser = argparse.ArgumentParser()
parser.add_argument("train_queries", help="Path of training queries, one per line")
parser.add_argument("train_gold", help="Path of gold hypernyms for the training queries")
parser.add_argument("--dev_size", "-d", type=int, default=50)
parser.add_argument("dir_out", help="Path of output directory")
args = parser.parse_args()

# Load data
queries, qtypes = utils.load_queries(args.train_queries, normalize=False)
gold = utils.load_hypernyms(args.train_gold, normalize=False)
assert len(queries) == len(gold)

# Shuffle indices
indices = list(range(len(queries)))
seed=91500
random.seed(seed)
random.shuffle(indices)

# Split
dev_indices = indices[:args.dev_size]
train_indices = indices[args.dev_size:]

# Write
if not os.path.exists(args.dir_out):
    os.mkdir(args.dir_out)
pq = os.path.join(args.dir_out, "train.queries.txt")
pg = os.path.join(args.dir_out, "train.gold.txt")
utils.write_queries_and_hypernyms(queries, gold, pq, pg, indices=train_indices)
pq = os.path.join(args.dir_out, "dev.queries.txt")
pg = os.path.join(args.dir_out, "dev.gold.txt")
utils.write_queries_and_hypernyms(queries, gold, pq, pg, indices=dev_indices)

