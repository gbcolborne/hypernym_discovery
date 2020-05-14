from copy import deepcopy
import numpy as np

""" Generate script for random hyperparameter search. """

lines = []
# Header
lines.append("#! /bin/bash")

# Constant part of our training command
base_cmd = "CUDA_VISIBLE_DEVICES=0 python run_ranker.py"
base_cmd += " --data_dir ../data/1A_random_split/ --lang en"
base_cmd += " --encoder_type xlm --encoder_name_or_path ../PretrainedModel_XLM_small_vocab"
base_cmd += " --do_train --evaluate_during_training"
base_cmd += " --per_gpu_eval_batch_size 512"
base_cmd += " --max_steps 200000 --logging_steps 5000 --save_steps 5000 --save_total_limit 1"

# Add flags
#base_cmd += " --freeze_query_encoder"
#base_cmd += " --freeze_cand_encoder"
#base_cmd += " --normalize_encodings"
base_cmd += " --subsample_positives"

# Set prefix for output directories
output_prefix = "Out1"

# Map short param names to long ones
param_key_to_name = {"bs":"per_gpu_train_batch_size",
                     "lr":"learning_rate",
                     "dp":"dropout_prob",
                     "ng":"nb_neg_samples",
                     "gn":"max_grad_norm"}

# Set param values we want to test
named_param_values = [("bs", ["16"]),
                      ("lr", ["1e-5"]),
                      ("dp", ["0.2"]),
                      ("ng", ["10"]),
                      ("gn", ["10"])]

# Generate all combinations
settings = [{}]
for key, values in named_param_values:
    tmp = []
    for setting in settings:
        for value in values:
            # Add value for this key to this setting
            new_setting = deepcopy(setting)
            new_setting[key] = value
            tmp.append(new_setting)
    settings = tmp[:]

# Remove specific combinations (e.g. to avoid using too much memory)
settings = [x for x in settings if not (x["ng"] == "32" and x["bs"] == "32")]
                      
# Take a random sample of settings
MAX_TESTS = 32
seed=91500
np.random.seed(seed)
np.random.shuffle(settings)
if len(settings) > MAX_TESTS:
    settings = settings[:MAX_TESTS]
    
# Make custom command for each test
for setting in settings:
    model_dir = "_".join([output_prefix, "Model"] + ["%s=%s"%(k,v) for k,v in setting.items()])
    eval_dir = "_".join([output_prefix, "Eval"] + ["%s=%s"%(k,v) for k,v in setting.items()])
    custom_cmd = base_cmd
    custom_cmd += " --model_dir %s" % model_dir
    custom_cmd += " --eval_dir %s" % eval_dir
    for k,v in setting.items():
        param_name = param_key_to_name[k]
        custom_cmd += " --%s %s" % (param_name, v)
    lines.append(custom_cmd + " ;")
    lines.append("rm -rf %s ;" % model_dir)

# Write commands
for line in lines:
    print(line)
