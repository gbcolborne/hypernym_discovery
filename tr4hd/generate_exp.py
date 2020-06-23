import random, string, argparse
from copy import deepcopy
import numpy as np
from cluster_utils import get_psub_command, get_base_command

""" Generate script for random hyperparameter search. """

# Constants
NOHUP = False    # Run with nohup
PSUB = True     # Run with psub
CUDA_DEVICES="0"
NB_TESTS = 100
SEED = None
DELIM = ";"

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None, help="Seed for RNGs (if you want to seed them)")
args = parser.parse_args()

# Seed RNGs
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

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
base_cmd += " --max_steps 25000"
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
                     "lw": "loss_weighting",
                     "bs": "per_gpu_train_batch_size",
                     "ng": "nb_neg_samples",
                     "sn": "smoothe_neg_sampling",
                     "ss": "pos_subsampling_factor",
                     "gn": "max_grad_norm",
                     "lr": "learning_rate",
                     "dp": "dropout_prob",
                     "wd": "weight_decay",
                     "wn": "weight_decay_norm",
                     }

# Set param values we want to test. For flags, use True or False. For args, use strings.
named_param_values = {"ea": ["single_q", "single_c", "bi_q", "bi_c"],
                      "fe": [True, False],
                      "ne": [True, False],
                      "tr": ["none", "scaling", "projection", "highway"],
                      "sf": ["dot", "spon"],
                      "ep": ["1e-1", "1e-3", "1e-5", "1e-7"],                      
                      "lf": ["nll", "bce", "nolog"],
                      "lw": ["none", "npos"],
                      "bs": ["8", "16", "32"],
                      "ng": ["4", "8", "16"],
                      "sn": [True, False],
                      "ss": ["0.0", "1.0"],
                      "gn": ["-1", "4", "8", "16"],                      
                      "lr": ["1e-3", "1e-4", "1e-5", "1e-6"],                      
                      "dp": ["0.0", "0.1", "0.2", "0.4"],
                      "wd": ["0", "1e-1", "1e-3", "1e-5", "1e-7"],
                      "wn": ["L1", "L2"],
                      }

def settings_are_valid(settings):
    # If encoder is frozen, there must be a transform, otherwise the model has no tunable params
    if settings["fe"] is True and settings["tr"] == "none":
        return False
    return True
    
# Generate random settings
while len(cmds) < NB_TESTS:
    uniq_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    model_dir = "_".join([output_prefix, "Model", uniq_name])
    eval_dir = "_".join([output_prefix, "Eval", uniq_name])
    cmd = base_cmd
    cmd += " --model_dir %s" % model_dir
    cmd += " --eval_dir %s" % eval_dir
    setting = {}
    for k,v in named_param_values.items():
        setting[k] = random.choice(v)
    if not settings_are_valid(setting):
        continue
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
        cmd += " > %s " % nohup_fn
    cmds.append((uniq_name, cmd))
    
# Write commands
for (test_name, cmd) in cmds:
    if PSUB:
        jobname = "%s_%s" % (output_prefix, test_name)
        psub_cmd = get_psub_command(jobname) 
        cmd = '%s "%s"' % (psub_cmd, cmd)
    print(cmd + " " + DELIM)
