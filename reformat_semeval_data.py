import argparse, os, shutil

"""
Reformat dataset of SemEval 2018 Task 9 (hypernym discovery). Write
one gold hypernym per line. Duplicate queries for each of their
hypernyms. Remove query types.
"""

parser = argparse.ArgumentParser()
parser.add_argument("dir_data", help="Path of dataset for SemEval 2018 Task 9")
parser.add_argument("--subtask", "-s", choices=["1A", "1B", "1C", "2A", "2B"], default="1A")
parser.add_argument("dir_out")
args = parser.parse_args()


subtask_names = {"1A":"1A.english", "1B":"1B.italian", "1C":"1C.spanish", "2A":"2A:medical", "2B":"2B.music"}
subtask_name = subtask_names[args.subtask]

if not os.path.exists(args.dir_out):
    os.mkdir(args.dir_out)

# Copy candidates
path_cand = os.path.join(args.dir_data, "vocabulary/{}.vocabulary.txt".format(subtask_name))
dst = os.path.join(args.dir_out, "candidates.txt".format(args.subtask))
shutil.copyfile(path_cand, dst)

# Reformat queries and gold hypernyms
for part in ["training", "trial", "test"]:
    # Load queries
    path_queries = os.path.join(args.dir_data, "{}/data/{}.{}.data.txt".format(part, subtask_name, part))
    with open(path_queries) as f:
        queries = []
        for line in f:
            elems = line.strip().split("\t")
            if len(elems)==2:
                query = elems[0]
                qtype = elems[1]
                queries.append(query)

    # Load gold hypernyms
    path_gold = os.path.join(args.dir_data, "{}/gold/{}.{}.gold.txt".format(part, subtask_name, part))
    with open(path_gold) as f:
        gold = []
        for line in f:
            if len(line.strip()):
                hyps = line.strip().split("\t")
                gold.append(hyps)
    
    assert len(queries) == len(gold)

    # Write queries and hypernyms, one per line, duplicating queries
    # for each of their hypernyms
    path_out_q = os.path.join(args.dir_out, "{}.queries.txt".format(part))
    path_out_g = os.path.join(args.dir_out, "{}.gold.txt".format(part))
    with open(path_out_q, 'w') as fq, open(path_out_g, 'w') as fg:
        for i in range(len(queries)):
            nb_hyps = len(gold[i])
            for j in range(nb_hyps):
                fq.write(queries[i]+"\n")
                fg.write(gold[i][j]+"\n")

