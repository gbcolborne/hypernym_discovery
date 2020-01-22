import argparse
import subprocess

""" Apply BPE to gold labeled data. First, replace tabs by newlines, then apply BPE, then join terms with tabs again. """

parser = argparse.ArgumentParser()
parser.add_argument("path_fastbpe")
parser.add_argument("gold_file")
parser.add_argument("output_file")
parser.add_argument("bpe_codes")
args = parser.parse_args()

# Replace tabs by newlines
path_tmp_in = args.gold_file + ".tmp"
with open(args.gold_file) as fi, open(path_tmp_in, "w") as fo:
    for line in fi:
        terms = line.strip().split("\t")
        for t in terms:
            fo.write("{}\n".format(t))
        fo.write("\n")

# Apply BPE
path_tmp_out = args.output_file + ".tmp"
cmd = [args.path_fastbpe, "applybpe", path_tmp_out, path_tmp_in, args.bpe_codes]
try:
    status = subprocess.run(cmd, check=True)
except CalledProcessError as e:
    raise e

# Load BPE, join with tabs and write
with open(path_tmp_out) as fi, open(args.output_file, 'w') as fo:
    terms = []
    for line in fi:
        line = line.strip()
        if len(line):
            terms.append(line)
        else:
            fo.write("{}\n".format("\t".join(terms)))
            terms = []

# Clean up
for path in path_tmp_in, path_tmp_out:
    cmd = ["rm", path]
    _ = subprocess.run(cmd)
