import os

""" Remove trailing @@ and space after tabs, resulting from applying BPE. """

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file")
args = parser.parse_args()

with open(input_file) as fi, open(output_file, 'w') as fo:
    for line in fi:
        line = line.replace("\t@@ ", "\t")
        fo.write(line)
