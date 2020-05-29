from copy import deepcopy
import numpy as np

""" Generate script for random hyperparameter search. """

lines = []
# Header
lines.append("#! /bin/bash")

# Constant part of our training command
base_cmd = "CUDA_VISIBLE_DEVICES=0 python run_spon.py"
base_cmd += " --data_dir ../data/1A_random_split/ --lang en"
base_cmd += " --encoder_type xlm --encoder_name_or_path ../PretrainedModel_XLM_small_vocab"
base_cmd += " --do_train --evaluate_during_training"
base_cmd += " --per_gpu_eval_batch_size 512"
base_cmd += " --max_steps 100000 --logging_steps 1000 --save_steps 1000 --save_total_limit 1"

# Set prefix for output directories
output_prefix = "Out_SPON"

# Map short param names to long ones
param_key_to_name = {"bs":"per_gpu_train_batch_size",
                     "lr":"learning_rate",
                     "dp":"dropout_prob",
                     "ng":"nb_neg_samples",
                     "gn":"max_grad_norm",
                     "ss":"pos_subsampling_factor",
                     "ol":"output_layer_type",
                     "ep":"spon_epsilon",
                     "iq":"iq_penalty",
                     "fe":"freeze_encoder",
                     "ne":"normalize_encodings",
                     "sn":"smoothe_neg_sampling",}

# Set param values we want to test. For flags, use True or False. For args, use strings.
named_param_values = {"bs": ["16"],
                      "lr": ["2e-5"],
                      "dp": ["0.0"],
                      "ng": ["10"],
                      "gn": ["-1"],
                      "ss": ["0.0"],
                      "ol": ["highway"],
                      "ep": ["1e-5"],
                      "iq": ["0.0"],
                      "fe": [False],
                      "ne": [False],
                      "sn": [False, True],}

# Generate all combinations
settings = [{}]
for key, values in named_param_values.items():
    tmp = []
    for setting in settings:
        for value in values:
            # Add value for this key to this setting
            new_setting = deepcopy(setting)
            new_setting[key] = value
            tmp.append(new_setting)
    settings = tmp[:]

# Take a random sample of settings
MAX_TESTS = 32
seed=91500
np.random.seed(seed)
np.random.shuffle(settings)
if len(settings) > MAX_TESTS:
    settings = settings[:MAX_TESTS]
    
# Make custom command for each test
for setting in settings:
    model_dir = "_".join([output_prefix, "Model"] + ["%s=%s"%(k,v) for k,v in setting.items() if len(named_param_values[k]) > 1])
    eval_dir = "_".join([output_prefix, "Eval"] + ["%s=%s"%(k,v) for k,v in setting.items() if len(named_param_values[k]) > 1])
    custom_cmd = base_cmd
    custom_cmd += " --model_dir %s" % model_dir
    custom_cmd += " --eval_dir %s" % eval_dir
    for k,v in setting.items():
        param_name = param_key_to_name[k]
        # Args
        if type(v) == str:
            custom_cmd += " --%s %s" % (param_name, v)
        # Flags
        else:
            if v is True:
                custom_cmd += " --%s" % (param_name)
    lines.append(custom_cmd + " ;")
    lines.append("rm -rf %s ;" % model_dir)

# Write commands
for line in lines:
    print(line)
