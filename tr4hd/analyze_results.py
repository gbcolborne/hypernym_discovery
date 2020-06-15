import os, glob, torch

pattern = "exp1_Eval_*"
eval_dirs = glob.glob(pattern)
settings = []
max_scores = []
best_steps = []

for eval_dir in eval_dirs:
    # Look for subdir containing results
    if not "csv" in os.listdir(eval_dir):
        continue
    
    # Look for evaluation results and test settings
    scores_path = os.path.join(eval_dir, "csv/eval_MAP.csv")
    model_dir = eval_dir.replace("_Eval_", "_Model_")
    args_path = os.path.join(model_dir, "training_args.bin")
    if not os.path.exists(scores_path) or not os.path.exists(args_path):
        continue

    # Load test settings
    settings.append(torch.load(args_path))

    # Load scores
    steps = []
    scores = []
    with open(scores_path) as f:
        # Skip header
        f.readline()
        for line in f:
            elems = line.strip().split(",")
            if len(elems) == 2:
                steps.append(elems[0])
                scores.append(float(elems[1]))

    # Find max score
    arg_max = -1
    max_score = -float("inf")
    for i in range(len(scores)):
        if scores[i] > max_score:
            max_score = scores[i]
            arg_max = i
    max_scores.append(scores[arg_max])
    best_steps.append(steps[arg_max])

# Analyze influence of hyperparams
hparam2scores = {}
for i in range(len(settings)):
    for k,v in settings[i]._get_kwargs():
        if k not in hparam2scores:
            hparam2scores[k] = []
        hparam2scores[k].append((v, max_scores[i]))
for k,v in hparam2scores.items():
    val2scores = {}
    for (val, score) in v:
        if val not in val2scores:
            val2scores[val] = []
        val2scores[val].append(score)
    if len(val2scores) == 1:
        continue
    if all([len(s) == 1 for v,s in val2scores.items()]):
        continue
    print("\n%s:" % k)
    for (val, scores) in sorted(val2scores.items(), key=lambda x:x[0], reverse=True):
        max_score = max(scores)
        avg_score = sum(scores)/len(scores)
        print(" - %s (n=%d): max=%f, avg=%f" % (val, len(scores), max_score, avg_score))

# Show best settings
arg_max = -1
max_score = -float("inf")
for i in range(len(max_scores)):
    if max_scores[i] > max_score:
        max_score =  max_scores[i]
        arg_max = i
print("\n\nBEST SETTINGS:")
print(" - Settings: %s" % settings[arg_max])
print(" - Max score: %s" % max_scores[arg_max])
print(" - Best step: %s" % best_steps[arg_max])
print(" - Eval dir: %s" % eval_dirs[arg_max])
