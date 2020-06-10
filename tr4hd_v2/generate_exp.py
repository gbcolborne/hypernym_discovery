import random, string
from copy import deepcopy
import numpy as np
from cluster_utils import get_psub_command, get_base_command

""" Generate script for random hyperparameter search. """

# User-defined constants
NOHUP = False    # Run with nohup
PSUB = False     # Run with psub
CUDA_DEVICES="0"
MAX_TESTS = 32
SEED = 91500

# Seed RNGs
random.seed(SEED)
np.random.seed(SEED)

# Initialize command template and list of commands
cmds = []
base_cmd = get_base_command()

# Constant part of our training command
base_cmd += " CUDA_VISIBLE_DEVICES=%s" % CUDA_DEVICES
if NOHUP:
    base_cmd += " nohup"
base_cmd += " python run_scorer.py"
base_cmd += " --data_dir ../data/1A_random_split/"
base_cmd += " --lang en"
base_cmd += " --encoder_type xlm"
base_cmd += " --encoder_name_or_path ../PretrainedModel_XLM_small_vocab"
base_cmd += " --do_train"
base_cmd += " --evaluate_during_training"
base_cmd += " --per_gpu_eval_batch_size 512"
base_cmd += " --max_steps 100000"
base_cmd += " --logging_steps 1000"
base_cmd += " --save_steps 1000"
base_cmd += " --save_total_limit 1"

# Set prefix for output directories
output_prefix = "exp1"

# Map short param names to long ones
param_key_to_name = {"ea": "encoding_arch",
                     "fe": "freeze_encoder",
                     "ne": "normalize_encodings",
                     "tr": "transform",
                     "sf": "score_fn",
                     "ep": "spon_epsilon",
                     "lf": "loss_fn",
                     "bs": "per_gpu_train_batch_size",
                     "ng": "nb_neg_samples",
                     "sn": "smoothe_neg_sampling",
                     "ss": "pos_subsampling_factor",
                     "gn": "max_grad_norm",
                     "lr": "learning_rate",
                     "dp": "dropout_prob",
                     "wd": "weight_decay",
                     }

# Set param values we want to test. For flags, use True or False. For args, use strings.
named_param_values = {"ea": ["single"], # "bi"
                      "fe": [False],
                      "ne": [False],
                      "tr": ["none"],   # "scaling", "projection", "highway"
                      "sf": ["dot"],    # "spon"
                      "ep": ["1e-5"],                      
                      "lf": ["nll"],    # "nllmod"
                      "bs": ["16"],
                      "ng": ["10"],
                      "sn": [False],
                      "ss": ["0.0"],
                      "gn": ["-1"],                      
                      "lr": ["2e-5"],                      
                      "dp": ["0.0"],
                      "wd": ["0"],
                      }

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
np.random.shuffle(settings)
if len(settings) > MAX_TESTS:
    settings = settings[:MAX_TESTS]
    
# Make custom command for each test
for setting in settings:
    uniq_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    model_dir = "_".join([output_prefix, "Model", uniq_name])
    eval_dir = "_".join([output_prefix, "Eval", uniq_name])
    if PSUB:
        jobname = "%s_%s" % (output_prefix, uniq_name)
        psub_cmd = get_psub_command(jobname) 
        cmd = psub_cmd + " " + base_cmd
    else:
        cmd = base_cmd
    cmd += " --model_dir %s" % model_dir
    cmd += " --eval_dir %s" % eval_dir
    for k,v in setting.items():
        param_name = param_key_to_name[k]
        # Args
        if type(v) == str:
            cmd += " --%s %s" % (param_name, v)
        # Flags
        else:
            if v is True:
                cmd += " --%s" % (param_name)
    if NOHUP:
        nohup_fn = "nohup_%s_%s.out" % (output_prefix, uniq_name)
        cmd += " > %s & " % nohup_fn
    else:
        cmd += " ;"
    cmds.append(cmd)
    
# Write commands
for cmd in cmds:
    print(cmd)
