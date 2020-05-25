#! /usr/bin/env python

"""
Train or evaluate ranker.
"""

from __future__ import absolute_import, division, print_function
import os, argparse, logging, json, random
import numpy as np
import torch
import torch.nn.functional as F
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
from transformers import AdamW
from sklearn.metrics import average_precision_score
from data_utils import make_train_set, make_dev_set, make_test_set, make_candidate_set, load_hd_data, rotate_checkpoints, get_missing_inputs
from SPONScorer import SPONScorer

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLMConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'xlm': (XLMConfig, XLMModel, XLMTokenizer),
}

RANKING_CUTOFF = 15


def load_model_and_tokenizer(opt, encoder_config, tokenizer_class):
    # Load training options
    train_opt = torch.load(os.path.join(opt.model_dir, 'training_args.bin'))

    # Add missing options to opt
    for k,v in train_opt._get_kwargs():
        if k not in opt or (k in opt and opt.__dict__[k] is None and train_opt.__dict__[k] is not None):
            opt.__dict__[k] = v
    
    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(opt.model_dir, do_lower_case=opt.do_lower_case)
    
    # Load model
    model = SPONScorer(opt, pretrained_encoder=None, encoder_config=encoder_config)
    model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'state_dict.pt')))
    model.to(opt.device)
    return model, tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def encode_batch(opt, model, tokenizer, batch, grad=False):
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
        encs = model.encode(inputs) 
    else:
        with torch.no_grad():
            encs = model.encode(inputs) 
    return encs


def encode_all_inputs(opt, model, tokenizer, inputs, grad=False, these_are_candidates=False, batch_size=128):
    """ Encode all inputs (candidates or queries). Return Tensor of encodings.
    Args:
    - opt
    - model
    - tokenizer
    - inputs: TensorDataset containing input_ids, nb_tokens

    """
    sampler = SequentialSampler(inputs)
    dataloader = DataLoader(inputs, sampler=sampler, batch_size=batch_size)
    all_encs = []
    for batch in tqdm(dataloader, desc="Encoding {}".format("candidates" if these_are_candidates else "queries"), leave=False):
        encs = encode_batch(opt, model, tokenizer, batch, grad=grad)
        all_encs.append(encs)
    all_encs = torch.cat(all_encs)
    return all_encs


def get_model_predictions(opt, model, tokenizer, query_inputs, cand_inputs):
    """
    Get model predictions for queries (without grad). Return scores.
    Args:
    - opt
    - model
    - tokenizer
    - query_inputs:  TensorDataset containing: input_ids, nb_tokens
    - cand_inputs:     TensorDataset containing: input_ids, nb_tokens

    """

    # multi-gpu eval
    if opt.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Set batch size
    opt.eval_batch_size = opt.per_gpu_eval_batch_size * max(1, opt.n_gpu)

    # Encode all queries
    query_encs = encode_all_inputs(opt, model, tokenizer, query_inputs, grad=False, these_are_candidates=False, batch_size=opt.eval_batch_size)

    # Make loader for candidates
    sampler = SequentialSampler(cand_inputs) 
    dataloader = DataLoader(cand_inputs, sampler=sampler, batch_size=opt.eval_batch_size)

    # Score all candidates for each query
    all_scores = [list() for _ in range(len(query_encs))]
    model.eval()
    batch_start = 0
    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        # Encode batch of candidates
        cand_encs = encode_batch(opt, model, tokenizer, batch, grad=False)
        batch_size = len(cand_encs)
        # Score this batch for all queries
        for query_ix in range(len(query_encs)):
            with torch.no_grad():
                scores = model({'query_enc': query_encs[query_ix]}, {'cand_encs':cand_encs}, convert_logits_to_probs=False)
            scores = scores.detach().cpu().numpy()
            all_scores[query_ix] += scores.tolist()
            
    # Convert logits to probabilities
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    y_probs = model.logits_to_probs(all_scores, dim=1)
    y_probs = y_probs.detach().cpu().numpy()
    return y_probs


def get_top_k_candidates_and_scores(scores):
    """ Get top-k candidates and scores.
    Args:
    - scores: numpy array of scores, shape (nb queries, nb candidates)

    """

    nb_queries = len(scores)
    top_candidates_and_scores = []
    for q in range(nb_queries):
        y_scores = scores[q,:]
        top_k_candidate_ids = np.argsort(y_scores).tolist()[-RANKING_CUTOFF:][::-1]
        top_k_scores = [y_scores[i] for i in top_k_candidate_ids]
        top_candidates_and_scores.append(list(zip(top_k_candidate_ids, top_k_scores)))
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
    query_inputs = make_test_set(opt, tokenizer, test_data, verbose=True)

    # Make dataset for candidate inputs
    cand_inputs = make_candidate_set(opt, tokenizer, test_data, verbose=True)

    # Get top k candidates and scores
    y_probs = get_model_predictions(opt, model, tokenizer, query_inputs, cand_inputs)
    logger.info("Extracting top {} candidates...".format(RANKING_CUTOFF))
    top_candidates_and_scores = get_top_k_candidates_and_scores(y_probs)
        
    # Write top k candidates and scores
    path_top_candidates = os.path.join(opt.eval_dir, "test_top_{}_candidates.tsv".format(RANKING_CUTOFF))
    path_top_scores = os.path.join(opt.eval_dir, "test_top_{}_scores.tsv".format(RANKING_CUTOFF))
    logger.info("  Writing top {} candidates for each query to {}".format(RANKING_CUTOFF, path_top_candidates))
    logger.info("  Writing top {} scores for each query to {}".format(RANKING_CUTOFF, path_top_scores))
    with open(path_top_candidates, 'w') as fc, open(path_top_scores, 'w') as fs:
        for topk in top_candidates_and_scores:
            last_ix = len(topk) - 1
            for i, (cand_id, score) in enumerate(topk):
                fc.write(test_data["candidates"][cand_id])
                fs.write("{:.5f}".format(score))
                if i == last_ix:
                    fc.write("\n")
                    fs.write("\n")
                else:
                    fc.write("\t")
                    fs.write("\t")
    return


def compute_loss(y_probs, targets, reduction=None):
    assert len(y_probs.size()) == 2
    assert len(targets.size()) == 1
    assert targets.size()[0] == y_probs.size()[0]
    if reduction is not None:
        assert reduction in ["sum", "mean"]
    target_probs = y_probs[:,targets]
    losses = -torch.log(target_probs)
    if reduction is None:
        return losses
    elif reduction == "sum":
        return torch.sum(losses)
    elif reduction == "mean":
        return torch.mean(losses)

def evaluate(opt, model, tokenizer, eval_data, cand_inputs):
    """ Evaluate model on labeled dataset (without grad). Return dictionary containing evaluation metrics.
    Args:
    - opt
    - model
    - tokenizer
    - eval_data: TensorDataset containing: input_ids, nb_tokens, padded_pos_cand_ids, nb_pos_cand_ids.
    - cand_inputs TensorDataset containing: input_ids, nb_tokens.

    """
    nb_queries = len(eval_data)
    nb_candidates = len(cand_inputs)
    padded_pos_cand_ids = eval_data.tensors[2].cpu().numpy()
    nb_pos_cand_ids = eval_data.tensors[3].cpu().numpy()

    # Get model predictions
    y_probs = get_model_predictions(opt, model, tokenizer, eval_data, cand_inputs)
    logger.debug("  Range(y_probs): {:.3f}-{:.3f}".format(np.min(y_probs), np.max(y_probs)))
    
    # Compute average precision scores
    ap_scores = []
    eval_iterator = trange(nb_queries, desc="Evaluating", leave=False)
    for i in eval_iterator:
        pos_cand_ids = padded_pos_cand_ids[i][:nb_pos_cand_ids[i]]
        y_true = np.zeros(nb_candidates)
        y_true[pos_cand_ids] = 1
        ap = average_precision_score(y_true=y_true, y_score=y_probs[i])
        ap_scores.append(ap)
    results = {"MAP": np.mean(ap_scores)}
    
    # Compute diversity of top-k predictions across queries
    top_candidates_and_scores = get_top_k_candidates_and_scores(y_probs)
    all_preds = []
    for i in range(nb_queries):
        for j in range(RANKING_CUTOFF):
            cand, score = top_candidates_and_scores[i][j]
            all_preds.append(cand)
    results["diversity"] = (len(set(all_preds)) - RANKING_CUTOFF) / (len(all_preds) - RANKING_CUTOFF)
    return results


def maybe_clip_grad(opt, model, optimizer):
    """ Clip grad norm in place. """
    if opt.max_grad_norm and opt.max_grad_norm > 0:
        if opt.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

        
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
        tb_writer = SummaryWriter(log_dir=opt.eval_dir)

    # Load training data
    train_data = load_hd_data(opt, 'train')


    # Make dev set
    dev_data = load_hd_data(opt, 'dev')
    dev_set = make_dev_set(opt, tokenizer, dev_data, verbose=True)

    # Make dataset for candidate inputs
    cand_inputs = make_candidate_set(opt, tokenizer, train_data, verbose=True)
    cand_input_ids = cand_inputs.tensors[0]
    cand_nb_tokens = cand_inputs.tensors[1]
    nb_candidates = len(cand_inputs)
    
    # Set batch size
    opt.train_batch_size = opt.per_gpu_train_batch_size * max(1, opt.n_gpu)

    # Make training set
    train_set = make_train_set(opt, tokenizer, train_data, seed=opt.seed, verbose=True)
    train_sampler = RandomSampler(train_set) if opt.local_rank == -1 else DistributedSampler(train_set)
    train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=opt.train_batch_size)

    # Set number of epochs and steps 
    if opt.max_steps > 0:
        total_steps = opt.max_steps # One step per batch of queries
        opt.num_train_epochs = opt.max_steps // len(train_dataloader) + 1
    else:
        total_steps = opt.num_train_epochs * len(train_dataloader) 
        
    # Prepare optimizer 
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=opt.adam_epsilon)
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

    # Train
    logger.info("***** Running training *****")
    logger.info("  Nb positive examples = %d", len(train_set))
    logger.info("  Nb candidates = %d", nb_candidates)
    logger.info("  Batch size = %d", opt.train_batch_size)
    logger.info("  Nb batches = %d", len(train_dataloader))
    logger.info("  Nb neg samples = {}".format(opt.nb_neg_samples))
    logger.info("  Nb Epochs = %d", opt.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d queries", opt.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel & distributed) = %d queries",
                   opt.train_batch_size * (torch.distributed.get_world_size() if opt.local_rank != -1 else 1))
    logger.info("  Total optimization steps (one per batch of queries) = %d", total_steps)
    global_step = 0
    training_loss, logging_loss = 0.0, 0.0
    training_grad_norm, logging_grad_norm = 0.0, 0.0
    train_iterator = trange(int(opt.num_train_epochs), desc="Epoch", disable=opt.local_rank not in [-1, 0])
    set_seed(opt.seed)  
    model.zero_grad()
    for epoch in train_iterator:
        if epoch > 0:
            # Reload training set to get fresh negative samples
            new_seed = opt.seed + epoch
            train_set = make_train_set(opt, tokenizer, train_data, seed=new_seed, verbose=False)
            train_sampler = RandomSampler(train_set) if opt.local_rank == -1 else DistributedSampler(train_set)
            train_dataloader = DataLoader(train_set, sampler=train_sampler, batch_size=opt.train_batch_size)
        epoch_iterator = tqdm(train_dataloader, desc="Step (batch)", leave=False, disable=opt.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Unpack batch
            query_input_ids = batch[0]
            query_nb_tokens = batch[1]
            cand_ids = batch[2]
            labels = batch[3]    
            bs = query_input_ids.size()[0]
            
            # Encode queries
            model.train()
            query_batch = (query_input_ids, query_nb_tokens)
            query_encs = encode_batch(opt, model, tokenizer, query_batch, grad=True)

            # Loop over queries
            scores = []
            for query_ix in range(bs):
                # Encode the candidates for this query
                cand_input_ids_sub = cand_input_ids[cand_ids[query_ix]]
                cand_nb_tokens_sub = cand_nb_tokens[cand_ids[query_ix]]
                cand_batch = (cand_input_ids_sub, cand_nb_tokens_sub)
                cand_encs = encode_batch(opt, model, tokenizer, cand_batch, grad=True)

                # Forward pass
                scores_sub = model({'query_enc': query_encs[query_ix]}, {'cand_encs':cand_encs}, convert_logits_to_probs=True)
                scores.append(scores_sub.unsqueeze(0))

            # Compute loss
            scores = torch.cat(scores, dim=0)
            loss = compute_loss(scores, labels, reduction='mean')
            if opt.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training

            # Backprop
            if opt.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Check norm of grad before clipping
            for _,param in model.named_parameters():
                if param.grad is not None:
                    training_grad_norm += torch.norm(param.grad, p=2).item()
                
            # Update
            maybe_clip_grad(opt, model, optimizer)
            optimizer.step()
            model.zero_grad()
            global_step += 1
            
            # Add loss scalar to total loss
            loss = loss.item()
            training_loss += loss

            # Check if we log loss and validation metrics
            if opt.local_rank in [-1, 0] and opt.logging_steps > 0 and global_step % opt.logging_steps == 0:
                logs = {}

                # Check if we validate on dev set
                if opt.local_rank == -1 and opt.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(opt, model, tokenizer, dev_set, cand_inputs)
                    for key, value in results.items():
                        eval_key = 'eval_{}'.format(key)
                        logs[eval_key] = value

                # Log loss, gradient norm
                loss_scalar = (training_loss - logging_loss) / opt.logging_steps
                logs['train_loss'] = loss_scalar 
                logging_loss = training_loss
                grad_norm = (training_grad_norm - logging_grad_norm) / opt.logging_steps
                logs['norm_w_grad'] = grad_norm
                logging_grad_norm = training_grad_norm

                # Log magnitude of model weights
                norm_w = 0.0
                for _,param in model.named_parameters():
                    norm_w += torch.norm(param.data, p=2).item()
                logs['norm_w'] = norm_w
                
                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                logger.info("  " + json.dumps({**logs, **{'step': global_step}}))
                    
            # Check if we save model checkpoint
            if opt.local_rank in [-1, 0] and opt.save_steps > 0 and global_step % opt.save_steps == 0:
                checkpoint_prefix = 'checkpoint'
                output_dir = os.path.join(opt.model_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                if not os.path.exists(output_dir) and opt.local_rank in [-1,0]:
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                logger.info("  Saving model checkpoint to %s", output_dir)
                tokenizer.save_pretrained(opt.model_dir)
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'state_dict.pt'))
                torch.save(opt, os.path.join(output_dir, 'training_args.bin'))
                rotate_checkpoints(opt.save_total_limit, opt.model_dir, checkpoint_prefix)
                    
            if opt.max_steps > 0 and global_step > opt.max_steps:
                epoch_iterator.close()
                break

        if opt.max_steps > 0 and global_step > opt.max_steps:
            train_iterator.close()
            break

    if opt.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, training_loss / global_step


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

    # Required if do_train or do_pred
    parser.add_argument("--eval_dir", default=None, type=str, required=False,
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
    parser.add_argument("--max_seq_length", default=32, type=int,
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
                        help="Set this flag if you are using an uncased model (warning: may also remove accents).")
    parser.add_argument("--output_layer_type", choices=['base', 'projection', 'highway'], required=False,
                        help="Output layer type")
    parser.add_argument("--spon_epsilon", type=float, default=1e-8,
                        help="Positive real value of epsilon in the SPON distance from satisfaction")
    
    parser.add_argument("--per_gpu_eval_batch_size", default=512, type=int,
                        help="Batch size (nb queries) per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size (nb queries) per GPU/CPU for training.")
    parser.add_argument("--nb_neg_samples", default=10, type=int, 
                        help=("Nb neg samples per positive example (for training only)."))
    parser.add_argument("--pos_subsampling_factor", type=float, default=0.0,
                        help="Real number between 0 and 1 that controls how aggressively we subsample positive examples during training")
    parser.add_argument("--freeze_encoder", action='store_true', help="Freeze weights of encoder during training.")
    parser.add_argument("--normalize_encodings", action='store_true', help="Normalize encodings")    
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.0, type=float,
                        help="Probability of zeroing out values in the query and candidate encodings.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=None, type=float, required=False,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
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
    parser.add_argument('--seed', type=int, default=91500,
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
    assert opt.do_train or opt.do_eval or opt.do_pred
    assert opt.logging_steps != 0
    assert opt.save_steps != 0
    assert opt.pos_subsampling_factor >= 0.0
    assert opt.pos_subsampling_factor <= 1.0
    if opt.do_train:
        if opt.encoder_name_or_path is None and opt.model_dir is None:
            raise ValueError("Either --encoder_name_or_path or model_dir must be specified if --do_train")
        if opt.encoder_name_or_path is None:
            path = os.path.join(opt.model_dir, 'state_dict.pt')
            if not os.path.exists(path):
                raise ValueError("--model_dir must contain a model if --encoder_name_or_path is not specified")
        elif opt.output_layer_type is None:
            msg = ("--output_layer_type must be specified unless you are re-training a model "
                   "(in which case use --model_dir rather than --encoder_name_or_path)")
            raise ValueError(msg)
    if opt.do_train or opt.do_pred:
        assert opt.eval_dir is not None
    if os.path.exists(opt.model_dir) and os.listdir(opt.model_dir) and opt.do_train and opt.encoder_name_or_path is not None and not opt.overwrite_model_dir:
        raise ValueError("Model directory ({}) already exists and is not empty. Use --overwrite_model_dir to overcome.".format(opt.model_dir))
    if (opt.do_train or opt.do_pred) and os.path.exists(opt.eval_dir) and os.listdir(opt.eval_dir):
        raise ValueError("Eval directory ({}) already exists and is not empty.".format(opt.eval_dir))
    opt.max_length = opt.max_seq_length
    if opt.encoder_type == 'xlm':
        opt.max_position_embeddings = opt.max_seq_length
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
    DEBUG = False
    level = None
    if opt.local_rank in [-1, 0]:
        if DEBUG:
            level = logging.DEBUG
        else:
            level = logging.INFO 
    else:
        level = logging.WARN
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = level)
    logger.warning("  Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    opt.local_rank, device, opt.n_gpu, bool(opt.local_rank != -1), opt.fp16)

    # Set seed
    set_seed(opt.seed)

    # Set up task
    task = "hyperdisco"
    num_labels = 2

    # Load encoder config
    opt.encoder_type = opt.encoder_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opt.encoder_type]
    if opt.encoder_config_name: 
        name_or_path = opt.encoder_config_name
    elif opt.do_train and opt.encoder_name_or_path is not None:
        name_or_path = opt.encoder_name_or_path
    else:
        # If we don't train and --encoder_name_or_path is not specified, we assume encoder config is in --model_dir
        name_or_path = opt.model_dir
    config = config_class.from_pretrained(name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=task,
                                          cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)
    
    # Training
    if opt.do_train:
        # Create model directory if needed
        if not os.path.exists(opt.model_dir) and opt.local_rank in [-1, 0]:
            os.makedirs(opt.model_dir)

        if opt.encoder_name_or_path is None:
            # Load previously trained model
            model, tokenizer = load_model_and_tokenizer(opt, config, tokenizer_class)
        else: 
            # Make sure only the first process in distributed training will download model 
            if opt.local_rank not in [-1, 0]:
                torch.distributed.barrier()  

            # Load pretrained encoder and tokenizer
            tokenizer = tokenizer_class.from_pretrained(opt.tokenizer_name if opt.tokenizer_name else opt.encoder_name_or_path,
                                                        do_lower_case=opt.do_lower_case,
                                                        cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)
            pretrained_encoder = model_class.from_pretrained(opt.encoder_name_or_path,
                                                             from_tf=bool('.ckpt' in opt.encoder_name_or_path),
                                                             config=config,
                                                             cache_dir=opt.encoder_cache_dir if opt.encoder_cache_dir else None)

            # Save config in model directory
            config.save_pretrained(opt.model_dir)
        
            # End of barrier
            if opt.local_rank == 0:
                torch.distributed.barrier()  
        
            # Initialize model
            model = SPONScorer(opt, pretrained_encoder=pretrained_encoder)
            model.to(opt.device)

        # Run training loop
        logger.info("  Training/evaluation parameters %s", opt)
        global_step, tr_loss = train(opt, model, tokenizer)
        logger.info("    global_step = %s, average loss = %s", global_step, tr_loss)

        # Save model and tokenizer
        if opt.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving model to %s", opt.model_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            tokenizer.save_pretrained(opt.model_dir)
            torch.save(model_to_save.state_dict(), os.path.join(opt.model_dir, 'state_dict.pt'))
            torch.save(opt, os.path.join(opt.model_dir, 'training_args.bin'))

    if opt.do_eval or opt.do_pred:
        model, tokenizer = load_model_and_tokenizer(opt, config, tokenizer_class)
        
    # Evaluation on dev set
    if opt.do_eval and opt.local_rank in [-1, 0]:
        # Make dev set
        dev_data = load_hd_data(opt, 'dev')
        dev_set = make_dev_set(opt, tokenizer, dev_data, verbose=True)

        # Make dataset for candidate inputs
        cand_inputs = make_candidate_set(opt, tokenizer, dev_data, verbose=True)

        # Load tokenizer and model
        eval_results = evaluate(opt, model, tokenizer, dev_set, cand_inputs)
        logger.info("***** Evaluation Results *****")
        for k, v in eval_results.items():
            print("{}: {}".format(k, v))
        
    # Prediction on test set
    if opt.do_pred and opt.local_rank in [-1, 0]:
        predict(opt, model, tokenizer)
        eval_results = None

    return eval_results



if __name__ == "__main__":

    results = main()
