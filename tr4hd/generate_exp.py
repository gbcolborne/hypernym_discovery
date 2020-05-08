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
base_cmd += " --per_query_nb_examples 50 --per_gpu_eval_batch_size 512"
base_cmd += " --max_steps 40000 --logging_steps 2000 --save_steps 2000 --save_total_limit 1"

# Add flags
base_cmd += " --freeze_query_encoder"
base_cmd += " --freeze_cand_encoder"
base_cmd += " --project_encodings --add_eye_to_init"

# Set prefix for output directories
output_prefix = "Out5"

# Map short param names to long ones
param_key_to_name = {"bs":"per_gpu_train_batch_size",
                     "lr":"learning_rate",
                     "dp":"dropout_prob",
                     "pr":"pos_ratio"}

# Set param values we want to test
named_param_values = [("bs", ["16", "32", "64"]),
                      ("lr", ["le-6", "1e-5", "1e-4", "1e-3"]),
                      ("dp", ["0.0", "0.1", "0.2"]),
                      ("pr", ["0.1", "0.2", "0.4"])]

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

# Take a random sample of settings
NB_TESTS = 32
seed=91500
np.random.seed(seed)
np.random.shuffle(settings)
settings=settings[:NB_TESTS]

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
