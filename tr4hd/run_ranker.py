#! /usr/bin/env python

"""
Train or evaluate ranker.
"""

from __future__ import absolute_import, division, print_function
import os, argparse, logging, json, random
import numpy as np
import torch
import torch.nn.functional as nnfunc
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME 
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import XLMConfig, XLMModel, XLMTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score
from data_utils import make_train_set, make_dev_set, make_test_set, make_candidate_set, load_hd_data, rotate_checkpoints, get_missing_inputs
from BiEncoderScorer import BiEncoderScorer

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLMConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlm': (XLMConfig, XLMModel, XLMTokenizer),
}

RANKING_CUTOFF = 15

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encode_batch(opt, model, tokenizer, batch, grad=False, these_are_candidates=False):
    """ Encode batch of queries or candidates. 
    Args:
    - opt
    - model
    - tokenizer
    - batch: tuple of input tensors (input_ids and nb_tokens)

    """

    batch = tuple(t.to(opt.device) for t in batch)
    input_ids = batch[0]
    nb_tokens = batch[1]
    inputs = {'input_ids':input_ids}
    inputs.update(get_missing_inputs(opt, input_ids, nb_tokens, tokenizer.lang2id[opt.lang]))
    if grad:
        encs = model.encode_candidates(inputs) if these_are_candidates else model.encode_queries(inputs)
    else:
        with torch.no_grad():
            encs = model.encode_candidates(inputs) if these_are_candidates else model.encode_queries(inputs)
    return encs
    
def encode_candidates(opt, model, tokenizer, candidate_inputs, grad=False, batch_size=128):
    """ Encode candidates. Return Tensor of candidate encodings.
    Args:
    - opt
    - model
    - tokenizer
    - candidate_inputs: TensorDataset containing input_ids, nb_tokens
    - grad: compute gradient
    - batch_size

    """
    sampler = SequentialSampler(candidate_inputs)
    dataloader = DataLoader(candidate_inputs, sampler=sampler, batch_size=batch_size)
    logger.info("  Num candidates = %d", len(candidate_inputs))
    logger.info("  Batch size = %d", batch_size)
    cand_encs = []
    for batch in tqdm(dataloader, desc="Encoding"):
        encs = encode_batch(opt, model, tokenizer, batch, grad=grad, these_are_candidates=True)
        cand_encs.append(encs)
    cand_encs = torch.cat(cand_encs)
    return cand_encs


def get_model_predictions(opt, model, tokenizer, query_inputs, cand_encs):
    """
    Get model predictions for queries (without grad). Return scores.
    Args:
    - opt
    - model
    - tokenizer
    - query_inputs: TensorDataset containing: input_ids, nb_tokens
    - cand_encs: Tensor containing all candidate encodings.

    """

    # multi-gpu eval
    if opt.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Set batch size
    opt.eval_batch_size = opt.per_gpu_eval_batch_size * max(1, opt.n_gpu)

    # Make loader for queries
    sampler = SequentialSampler(query_inputs) 
    dataloader = DataLoader(query_inputs, sampler=sampler, batch_size=opt.eval_batch_size)

    # Start eval
    logger.info("  Nb queries = %d", len(query_inputs))
    logger.info("  Nb candidates = %d", len(cand_encs))
    logger.info("  Nb batches = %d", opt.eval_batch_size)
    all_y_probs = []
    model.eval()
    for batch in tqdm(dataloader, desc="Predicting"):
        y_probs = None
        # Encode queries
        query_encs = encode_batch(opt, model, tokenizer, batch, grad=False, these_are_candidates=False)
        for cand_id in range(len(cand_encs)):
            with torch.no_grad():
                scores = model({'query_encs': query_encs}, {'cand_encs':cand_encs[cand_id]})
            if y_probs is None:
                y_probs = scores.detach().cpu().numpy()
            else:
                y_probs = np.append(y_probs, scores.detach().cpu().numpy(), axis=0)
        all_y_probs.append(y_probs)
    return all_y_probs

def get_top_k_candidates_and_scores(scores):
    """ Get top-k candidates and scores.
    Args:
    - scores: list of list of scores (one list per query). 

    """

    nb_queries = len(scores)
    top_candidates_and_scores = []
    for q in range(nb_queries):
        y_scores = scores[q,1]
        top_k_candidate_ids = np.argsort(y_scores).tolist()[-RANKING_CUTOFF:][::-1]
        tok_k_scores = [y_scores[i] for i in top_k_candidate_ids]
        top_candidates_and_scores.append(zip(top_k_candidate_ids, top_k_scores))
    return top_candidates_and_scores


def predict(opt, model, tokenizer):
    """
    Get model predictions for test queries (without grad), and write 15 highest ranked candidates per query.
    Args:
    - opt
    - model
    - tokenizer

    """

    # Make output dir if necessary
    if not os.path.exists(opt.eval_dir) and opt.local_rank in [-1, 0]:
        os.makedirs(opt.eval_dir)

    # Make test set
    test_data = load_hd_data(opt, 'test')
    query_inputs = make_test_set(opt, tokenizer, test_data)

    # Make dataset for candidate inputs
    cand_inputs = make_candidate_set(opt, tokenizer, test_data)

    # Encode candidates
    cand_encs = encode_candidates(opt, model, tokenizer, cand_inputs, grad=False, batch_size=128)

    # Get top k candidates and scores
    y_probs = get_model_predictions(opt, model, tokenizer, query_inputs, cand_encs)
    top_candidates_and_scores = get_top_k_candidates_and_scores(y_probs)
        
    # Write top k candidates and scores
    path_top_candidates = os.path.join(opt.eval_dir, "test_top_{}_candidates.tsv".format(RANKING_CUTOFF))
    path_top_scores = os.path.join(opt.eval_dir, "test_top_{}_scores.tsv".format(RANKING_CUTOFF))
    logger.info("Writing top {} candidates for each query to {}".format(RANKING_CUTOFF, path_top_candidates))
    logger.info("Writing top {} scores for each query to {}".format(RANKING_CUTOFF, path_top_scores))
    with open(path_top_candidates, 'w') as fc, open(path_top_scores, 'w') as fs:
        for i, topk in enumerate(top_candidates_and_scores):
            fc.write("{}\n".format("\t".join([candidates[c] for (c,s) in topk])))
            fs.write("{}\n".format("\t".join(["{:.5f}".format(s) for (c,s) in topk])))
            query = queries[i]
            topk_string = ', '.join(["('{}',{:.5f})".format(candidates[c],s) for (c,s) in topk])
            logger.info("{}. Top candidates for '{}': {}".format(i+1, query, topk_string))
    return

def compute_loss(inputs, targets):
    return nnfunc.binary_cross_entropy(inputs, targets)

def evaluate(opt, model, eval_data, cand_inputs):
    """ Evaluate model on labeled dataset (without grad). Return dictionary containing average loss and evaluation metrics.
    Args:
    - opt
    - model
    - eval_data: TensorDataset containing: input_ids, nb_tokens, candidate_ids, labels.
    - cand_inputs TensorDataset containing: input_ids, nb_tokens.

    """

    nb_queries = len(eval_data)
    nb_candidates = len(cand_inputs)
    logger.info("***** Running evaluation *****")
    logger.info("  Nb queries: {}".format(nb_queries))
    logger.info("  Nb candidates: {}".format(nb_candidates))

    # Encode candidates
    model.eval()
    cand_encs = encode_candidates(opt, model, tokenizer, cand_inputs, grad=False, batch_size=128)

    # Get model predictions
    y_probs = get_model_predictions(opt, model, tokenizer, eval_data, cand_encs)
    y_probs = torch.tensor(y_probs, dtype=torch.float32)

    # Get labels
    y_true = eval_data.tensors[5]
    
    # Compute loss
    loss = compute_loss(y_probs, y_true)
    total_loss = loss.item()
    avg_loss = total_loss / (nb_queries * nb_candidates)
    results = {'avg_loss': avg_loss}

    # Compute evaluation metrics 
    ap_scores = [] # average precision
    for i in range(nb_queries):
        ys = y_probs[i,1]
        yt = y_true[i]
        ap = average_precision_score(y_true=yt, y_score=ys)
        ap_scores.append(ap)

    # Compute mean average precision
    MAP = np.mean(ap_scores)
    results["MAP"] = MAP
                
    logger.info("***** Results *****")
    logger.info("  MAP: {}".format(MAP))
    logger.info("  loss: {}".format(avg_loss))
    return results

def train(opt, model, tokenizer):
    """
    Run training on training set, with validation on dev set after each epoch. Return model as well as a dict containing losses and scores after each epoch.
    Args:
    - opt
    - model
    - tokenizer

    """

    # Create writer for tensorboard data
    if opt.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Make training set
    train_data = load_hd_data(opt, 'train')
    train_set = make_train_set(opt, tokenizer, train_data)

    # Make dev set
    dev_data = load_hd_data(opt, 'dev')
    dev_set = make_dev_set(opt, tokenizer, dev_data)

    # Make dataset for candidate inputs
    cand_inputs = make_candidate_set(opt, tokenizer, train_data)
    cand_input_ids = cand_inputs.tensors[0]
    cand_nb_tokens = cand_inputs.tensors[1]
    nb_candidates = len(cand_inputs)
    
    # Set batch size
    opt.train_batch_size = opt.per_gpu_train_batch_size * max(1, opt.n_gpu)

    # Make data loader for training data which randomly samples queries
    train_sampler = RandomSampler(train_set) if opt.local_rank == -1 else DistributedSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=opt.train_batch_size)

    # Set number of epochs and steps 
    if opt.max_steps > 0:
        total_steps = opt.max_steps # One step per batch of queries
        total_sub_steps = total_steps * opt.per_query_nb_examples // opt.gradient_accumulation_steps # One substep per candidate per batch of queries
        opt.num_train_epochs = opt.max_steps // (len(train_dataloader) // opt.gradient_accumulation_steps) + 1
    else:
        total_steps = opt.num_train_epochs * len(train_dataloader) 
        total_sub_steps = total_steps * opt.per_query_nb_examples // opt.gradient_accumulation_steps
        
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)
    # Scheduler will only step once per batch of queries
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_steps, num_training_steps=total_steps)
    if opt.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if opt.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if opt.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Nb queries = %d", len(train_set))
    logger.info("  Nb candidates = %d", nb_candidates)
    logger.info("  Batch size (nb queries) = %d", opt.train_batch_size)
    logger.info("  Nb candidates evaluated per query = %d", opt.per_query_nb_examples)
    logger.info("  Nb Epochs = %d", opt.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", opt.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d queries",
                   opt.train_batch_size * opt.gradient_accumulation_steps * (torch.distributed.get_world_size() if opt.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", opt.gradient_accumulation_steps)
    logger.info("  Total optimization steps (one per batch of queries) = %d", total_steps)
    logger.info("  Total optimization steps (one per candidate per batch of queries) = %d", total_sub_steps)    
    global_step = 0
    global_sub_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(opt.num_train_epochs), desc="Epoch", disable=opt.local_rank not in [-1, 0])
    set_seed(opt.seed)  
    model.zero_grad()
    for _ in train_iterator:

        # Uncomment to reload train set, so that we get new negative samples
        # train_set = make_train_set(opt, tokenizer, train_data)
        # train_sampler = RandomSampler(train_set) if opt.local_rank == -1 else DistributedSampler(train_set)
        # train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=opt.train_batch_size)
 
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=opt.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Unpack batch
            query_input_ids = batch[0]
            query_nb_tokens = batch[1]
            cand_ids = batch[2]
            labels = batch[3]    

            # Encode queries
            model.train()
            query_batch = (query_input_ids, query_nb_tokens)
            query_encs = encode_batch(opt, model, tokenizer, query_batch, grad=True, these_are_candidates=False)
            
            # Iterate over candidate indices, accumulate gradients
            for sub_step, cand_ix in enumerate(range(opt.per_query_nb_examples)):
                # Encode the <batch_size> candidates at this candidate index
                cand_ids_sub = cand_ids[:,cand_ix]
                cand_input_ids_sub = cand_input_ids[cand_ids_sub]
                cand_nb_tokens_sub = cand_nb_tokens[cand_ids_sub]
                cand_batch = (cand_input_ids_sub, cand_nb_tokens_sub)
                cand_encs = encode_batch(opt, model, tokenizer, cand_batch, grad=True, these_are_candidates=True)

                # Forward pass on <batch_size> pairs of (query, candidate) encodings
                scores = model({'query_encs': query_encs}, {'cand_encs':cand_encs})

                # Compute loss
                labels_sub = labels[:,cand_ix]
                sub_loss = compute_loss(scores, labels_sub)

                print(sub_loss)
                sub_loss = sub_loss / opt.per_query_nb_examples

                if opt.n_gpu > 1:
                    sub_loss = sub_loss.mean() # mean() to average on multi-gpu parallel training
                if opt.gradient_accumulation_steps > 1:
                    sub_loss = sub_loss / opt.gradient_accumulation_steps

                # Backprop
                if opt.fp16:
                    with amp.scale_loss(sub_loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    sub_loss.backward(retain_graph=True)
                # We retained the graph to not free the sub-graph for the forward pass of the query encoder, which we want to re-use. Free the candidate encodings sub-graph.
                cand_encs = cand_encs.detach()
                
                tr_loss += sub_loss                
                global_sub_step += 1
                
            # Check if we update or accumulate gradient
            if (step + 1) % opt.gradient_accumulation_steps == 0:
                # Clip grad
                if opt.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

                # Update model 
                optimizer.step()
                model.zero_grad()
                scheduler.step()  
                global_step += 1

            # Check if we log loss and validation metrics
            if opt.local_rank in [-1, 0] and opt.logging_steps > 0 and global_step % opt.logging_steps == 0:
                logs = {}

                # Check if we validate on dev set
                if opt.local_rank == -1 and opt.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(opt, model, dev_set, cand_inputs)
                    for key, value in results.items():
                        eval_key = 'eval_{}'.format(key)
                        logs[eval_key] = value

                # Log loss on training set and learning rate
                loss_scalar = (tr_loss - logging_loss) / opt.logging_steps
                logging_loss = tr_loss
                learning_rate_scalar = scheduler.get_lr()[0]
                logs['learning_rate'] = learning_rate_scalar
                logs['loss'] = loss_scalar
                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                    # logger.info("  " + json.dumps({**logs, **{'step': global_step}}))
                    
            # Check if we save model checkpoint
            if opt.local_rank in [-1, 0] and opt.save_steps > 0 and global_step % opt.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                output_dir = os.path.join(opt.model_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir) and opt.local_rank in [-1,0]:
                    os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'state_dict.pkl'))
            torch.save(opt, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            rotate_checkpoints(opt.save_total_limit, opt.model_dir, checkpoint_prefix)
                    
            if opt.max_steps > 0 and global_step > opt.max_steps:
                epoch_iterator.close()
                break

        if opt.max_steps > 0 and global_step > opt.max_steps:
            train_iterator.close()
            break

    if opt.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step





                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="Language of the dataset")
    parser.add_argument("--encoder_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="Path of directory where model and tokenizer are saved (if do_train) or loaded (if do_eval or do_pred)")
    parser.add_argument("--eval_dir", default=None, type=str, required=True,
                        help="The output directory where the predictions and evaluation results will be written.")

    # Required if do_train
    parser.add_argument("--encoder_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained encoder or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    ## Other parameters
    parser.add_argument("--encoder_config_name", default="", type=str,
                        help="Pretrained encoder config name or path if not the same as encoder_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as encoder_name")
    parser.add_argument("--encoder_cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models (encoders) downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pred", action="store_true",
                        help="Whether to run prediction on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation on dev set during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--freeze_cand_encoder", action='store_true',
                        help="Freeze weights of candidate encoder.")
    parser.add_argument("--per_query_nb_examples", default=50, type=int, 
                        help=("Nb candidates evaluated per query in a batch. "
                              "During training, nb negative examples is obtained by subtracting "
                              "the number of positive examples for a given query."))
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size (nb queries) per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size (nb queries) per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the model_dir, does not delete by default')
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_model_dir', action='store_true',
                        help="Overwrite the content of the directory containing the model")
    parser.add_argument('--overwrite_eval_dir', action='store_true',
                        help="Overwrite the content of the directory containing the evaluation results or predictions")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    opt = parser.parse_args()

    # Check args
    if opt.do_train:
        if opt.encoder_name_or_path is None:
            raise ValueError("--encoder_name_or_path must be specified if --do_train")
    if os.path.exists(opt.model_dir) and os.listdir(opt.model_dir) and opt.do_train and not opt.overwrite_model_dir:
        raise ValueError("Model directory ({}) already exists and is not empty. Use --overwrite_model_dir to overcome.".format(opt.model_dir))
    if os.path.exists(opt.eval_dir) and os.listdir(opt.eval_dir) and not opt.overwrite_eval_dir:
        raise ValueError("Eval directory ({}) already exists and is not empty. Use --overwrite_eval_dir to overcome.".format(opt.eval_dir))

    # Setup distant debugging if needed
    if opt.server_ip and opt.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(opt.server_ip, opt.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if opt.local_rank == -1 or opt.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")
        opt.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        opt.n_gpu = 1
    opt.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    opt.local_rank, device, opt.n_gpu, bool(opt.local_rank != -1), opt.fp16)

    # Set seed
    set_seed(opt.seed)

    # Set up task
    task = "hyperdisco"
    num_labels = 2
    opt.MAX_CANDIDATE_LENGTH = 20
    opt.MAX_QUERY_LENGTH = 20
    

    # Training
    if opt.do_train:
        # Create model directory if needed
        if not os.path.exists(opt.model_dir) and opt.local_rank in [-1, 0]:
            os.makedirs(opt.model_dir)

        # Make sure only the first process in distributed training will download model & vocab
        if opt.local_rank not in [-1, 0]:
            torch.distributed.barrier()  

        # Load pretrained encoder and tokenizer
        opt.encoder_type = opt.encoder_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[opt.encoder_type]
        config = config_class.from_pretrained(opt.encoder_config_name if opt.encoder_config_name else opt.encoder_name_or_path,
                                              num_labels=num_labels,
                                              finetuning_task=task,
                                              cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)
        tokenizer = tokenizer_class.from_pretrained(opt.tokenizer_name if opt.tokenizer_name else opt.encoder_name_or_path,
                                                    do_lower_case=opt.do_lower_case,
                                                    cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)
        pretrained_encoder = model_class.from_pretrained(opt.encoder_name_or_path,
                                                         from_tf=bool('.ckpt' in opt.encoder_name_or_path),
                                                         config=config,
                                                         cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)
    
        # End of barrier
        if opt.local_rank == 0:
            torch.distributed.barrier()  

        # Initialize model
        model = BiEncoderScorer(opt, pretrained_encoder)
        model.to(opt.device)

        # Run training loop
        logger.info("Training/evaluation parameters %s", opt)
        global_step, tr_loss = train(opt, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save model and tokenizer
        if opt.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("Saving model to %s", opt.model_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            tokenizer.save_pretrained(opt.model_dir)
            torch.save(model_to_save.state_dict(), os.path.join(opt.model_dir, 'state_dict.pkl'))
            torch.save(opt, os.path.join(opt.model_dir, 'training_args.bin'))

    if opt.do_eval or opt.do_pred:
        # Load tokenizer
        tokenizer = tokenizer_class.from_pretrained(opt.tokenizer_name if opt.tokenizer_name else opt.encoder_name_or_path,
                                                    do_lower_case=opt.do_lower_case,
                                                    cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)

        # Load model
        model = BiEncoderScorer(opt, None)
        model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'state_dict.pkl')))
        model.to(opt.device)

    # Evaluation on dev set
    if opt.do_eval and opt.local_rank in [-1, 0]:
        # Make dev set
        dev_data = load_hd_data(opt, 'dev')
        dev_set = make_dev_set(opt, tokenizer, dev_data)

        # Make dataset for candidate inputs
        cand_inputs = make_candidate_set(opt, tokenizer, dev_data)

        # Load tokenizer and model
        logger.info("Evaluate model on dev set")
        eval_results = evaluate(opt, model_to_eval, dev_set, cand_inputs)

    # Prediction on test set
    if opt.do_pred and opt.local_rank in [-1, 0]:
        predict(opt, model, tokenizer)
        eval_results = None

    return eval_results



if __name__ == "__main__":

    results = main()
