import argparse, random, os

""" Split training set into a random train/dev split. """

parser = argparse.ArgumentParser()
parser.add_argument("train_queries", help="Path of training queries, one per line")
parser.add_argument("train_gold", help="Path of training gold hypernyms, one per line")
parser.add_argument("--dev_size", "-d", type=int, default=50)
parser.add_argument("dir_out", help="Path of output directory")
args = parser.parse_args()

# Load data
queries = [x.strip() for x in open(args.train_queries)]
gold = [x.strip() for x in open(args.train_gold)]
assert len(queries) == len(gold)
data = {}
for k,v in zip(queries, gold):
    if k not in data:
        data[k] = []
    data[k].append(v)
    
# Shuffle queries
uniq_queries = list(set(queries))
seed=91500
random.seed(seed)
random.shuffle(uniq_queries)

# Split
dev_queries = uniq_queries[:args.dev_size]
train_queries = uniq_queries[args.dev_size:]

# Write
if not os.path.exists(args.dir_out):
    os.mkdir(args.dir_out)
pq = os.path.join(args.dir_out, "train.queries.txt")
pg = os.path.join(args.dir_out, "train.gold.txt")
with open(pq, 'w') as fq, open(pg, 'w') as fg:
    for q in train_queries:
        for g in data[q]:
            fq.write(q+"\n")
            fg.write(g+"\n")
pq = os.path.join(args.dir_out, "dev.queries.txt")
pg = os.path.join(args.dir_out, "dev.gold.txt")
with open(pq, 'w') as fq, open(pg, 'w') as fg:
    for q in dev_queries:
        for g in data[q]:
            fq.write(q+"\n")
            fg.write(g+"\n")
