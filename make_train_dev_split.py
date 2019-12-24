import argparse, random, os

parser = argparse.ArgumentParser()
parser.add_argument("path_input", type=str, help="path of text file containing corpus")
parser.add_argument("dev_size", type=int, help="nb lines to put in dev set")
args = parser.parse_args()

# Check input
args.dir_input, args.path_input = os.path.split(os.path.abspath(args.path_input))
args.lang = args.path_input.split(".")[0]

# Count lines
with open(args.path_input) as f:
    nb_lines = sum(1 for line in f)

# Sample indices of dev set
assert args.dev_size <= nb_lines
ix = list(range(nb_lines))
random.shuffle(ix)
dev_ix = set(ix[:args.dev_size])
od = os.path.join(args.dir_input, "{}.dev".format(args.lang))
ot = os.path.join(args.dir_input, "{}.train".format(args.lang))
with open(args.path_input) as fin, open(od, 'w') as fod, open(ot, 'w') as fot:
    for i,line in enumerate(fin):
        if i in dev_ix:
            fod.write(line)
        else:
            fot.write(line)
    
