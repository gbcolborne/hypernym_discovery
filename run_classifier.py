# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Finetuning the library models for sequence classification on the
hypernym discovery dataset for SemEval 2018 Task 9."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import re
import shutil
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
    
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer,
                          AlbertConfig,
                          AlbertForSequenceClassification, 
                          AlbertTokenizer,
                          )
from transformers.data.processors.utils import InputFeatures
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, average_precision_score
from utils import load_hypernyms

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir) and args.local_rank in [-1,0]:
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    _rotate_checkpoints(args, checkpoint_prefix)
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def get_model_predictions(args, model, eval_dataset):
    """Run prediction on dataset sequentially. Return predicted class
    probabilities of examples in original order, as well the true
    labels (if available), and the average loss.

    """

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Set batch size
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Make loader
    eval_sampler = SequentialSampler(eval_dataset) # MUST BE SEQUENTIAL!
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Start eval
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    y_probs = None
    y_true = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if y_probs is None:
            y_probs = logits.detach().cpu().numpy()
            y_true = inputs['labels'].detach().cpu().numpy()
        else:
            y_probs = np.append(y_probs, logits.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, inputs['labels'].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    return y_probs, y_true, eval_loss


def evaluate(args, model, tokenizer, prefix=""):
    """ Evaluate on dev set. """

    # Make output dir if necessary
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        

    # Get dev data
    dev_data = load_and_cache_dataset(args, tokenizer, 'dev')
    dev_queries = dev_data["queries"]
    dev_query_token_ids = dev_data["query_token_ids"]
    candidate_token_ids = dev_data["candidate_token_ids"]
    candidates = list(dev_data["candidate2id"].keys())
    candidate_ids = list(dev_data["candidate2id"].values())
    dev_pos_candidate_ids = dev_data["gold_hypernym_candidate_ids"]
    
    
    logger.info("***** Running evaluation *****")
    logger.info("  Nb queries: {}".format(len(dev_queries)))
                
    # Accumulate average precision scores
    ap_scores = []

    # Loop over queries
    total_eval_loss = 0.0
    nb_queries = len(dev_queries)
    for i in range(nb_queries):
        # Create a dataset for this query and all the candidates
        query_token_ids = dev_query_token_ids[i]
        candidate_labels = [0] * len(candidate_ids)
        for candidate_id in dev_pos_candidate_ids[i]:
            candidate_labels[candidate_id] = 1
        eval_dataset = make_dataset(tokenizer,
                                    [query_token_ids],
                                    candidate_token_ids,
                                    [candidate_ids],
                                    candidate_labels=[candidate_labels],
                                    max_length=args.max_seq_length,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True)
        # Evaluate model on dataset
        logger.info("  *** Running evaluation on query {} ('{}') ***".format(i, dev_queries[i]))
        y_probs, y_true, eval_loss = get_model_predictions(args, model, eval_dataset)
        total_eval_loss += eval_loss
        y_score = y_probs[:,1]
        ap = average_precision_score(y_true=y_true, y_score=y_score)
        ap_scores.append(ap)

    # Compute mean average precision
    MAP = np.mean(ap_scores)
    loss = total_eval_loss/nb_queries
                
    logger.info("***** Results *****")
    logger.info("  MAP: {}".format(MAP))
    logger.info("  loss: {}".format(loss))
    return {"MAP":MAP, "loss":loss}


def predict(args, model, tokenizer):
    """ Run prediction on test set. """

    # Make output dir if necessary
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Get test data
    test_data = load_and_cache_dataset(args, tokenizer, 'test')
    test_queries = test_data["queries"]
    test_query_token_ids = test_data["query_token_ids"]
    candidate_token_ids = test_data["candidate_token_ids"]
    candidates = list(test_data["candidate2id"].keys())
    candidate_ids = list(test_data["candidate2id"].values())

    # Check if gold is available
    gold_available = "gold_hypernym_candidate_ids" in test_data and test_data["gold_hypernym_candidate_ids"] is not None
    test_pos_candidate_ids = test_data["gold_hypernym_candidate_ids"] if gold_availble else None

    # Write and log top k candidates for each test query. 
    ranking_cutoff = 15

    logger.info("***** Running prediction *****")
    logger.info("  Nb queries: {}".format(len(test_queries)))
    logger.info("  Ranking cutoff: {}".format(ranking_cutoff))
    if gold_available:
        logger.info("  Evaluating ranking of candidates wrt gold hypernyms")
    else:
        logger.info("  NOT evaluating ranking of candidates (gold hypernyms not available)")
    
    # Accumulate average precision scores (if gold is available)
    ap_scores = []

    # Loop over queries
    total_test_loss = 0.0
    nb_queries = len(test_queries)
    for i in range(nb_queries):
        # Create a dataset for this query and all the candidates
        query_token_ids = test_query_token_ids[i]
        candidate_labels = [0] * len(candidate_ids)
        for candidate_id in test_pos_candidate_ids[i]:
            candidate_labels[candidate_id] = 1
        eval_dataset = make_dataset(tokenizer,
                                    [query_token_ids],
                                    candidate_token_ids,
                                    [candidate_ids],
                                    candidate_labels=[candidate_labels],
                                    max_length=args.max_seq_length,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True)
        logger.info(" *** Running prediction on query {} ('{}') ***".format(i, test_queries[i]))                
        y_probs, y_true, test_loss = get_model_predictions(args, model, eval_dataset)
        total_test_loss += test_loss

        # Get top k candidates and their scores
        y_scores = y_probs[:,1]
        top_k_candidate_ids = np.argsort(y_scores).tolist()[-ranking_cutoff:][::-1]
        tok_k_scores = [y_scores[i] for i in top_k_candidate_ids]
        top_candidates_and_scores.append(zip(top_k_candidate_ids, top_k_scores))
        
        # Evalute ranking if gold hypernyms are available
        if gold_available:
            y_score = y_probs[:,1]
            ap = average_precision_score(y_true=y_true, y_score=y_score)
            ap_scores.append(ap)

        # FOR DEBUGGING
        logger.warning("  STOPPING FOR DEBUGGING PURPOSES")
        break
    
    # Compute average loss
    loss = total_test_loss/nb_queries
    results["loss"] = loss
    logger.info("***** Results *****")
    logger.info("  loss: {}".format(loss))

    # Compute mean average precision if gold hypernyms were available
    if gold_available:
        MAP = np.mean(ap_scores)
        results["MAP"] = MAP
        logger.info("  MAP: {}".format(MAP))    
    
    # Write top k candidates and scores
    path_top_candidates = os.path.join(args.output_dir, "test_top_{}_candidates.tsv".format(ranking_cutoff))
    path_top_scores = os.path.join(args.output_dir, "test_top_{}_scores.tsv".format(ranking_cutoff))
    logger.info("Writing top {} candidates for each query to {}".format(ranking_cutoff, path_top_candidates))
    logger.info("Writing top {} scores for each query to {}".format(ranking_cutoff, path_top_scores))
    with open(path_top_candidates, 'w') as fc, open(path_top_scores, 'w') as fs:
        for i, topk in enumerate(top_candidates_and_scores):
            fc.write("{}\n".format("\t".join([c for (c,s) in topk])))
            fs.write("{}\n".format("\t".join(["{:.5f}".format(s) for (c,s) in topk])))
            query = test_queries[i]
            topk_string = ', '.join(["('{}',{:.5f})".format(c,s) for (c,s) in topk])
            logger.info("{}. Top candidates for '{}': {}".format(i+1, query, topk_string))
    
    # Write average precision of each query
    if gold_available:
        output_eval_file = os.path.join(args.output_dir, "test_average_precision.txt")
        logger.info("  Writing average precision scores in {}".format(output_eval_file))
        with open(output_eval_file, "w") as writer:
            for ap in ap_scores:
                writer.write("{:.5f}\n".format(ap))

    return results


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def tokenize_strings(tokenizer, strings, max_length):
    """Given a tokenizer and a list of strings, tokenize strings and
    return tokens and token IDs.

    """
    all_tokens = []
    all_token_ids = []
    for string in strings:
        tokens = tokenizer.tokenize(string)
        token_ids = tokenizer.encode(tokens,
                                     add_special_tokens=False,
                                     max_length=max_length,
                                     pad_to_max_length=False)
        all_tokens.append(tokens)
        all_token_ids.append(token_ids)
    return all_tokens, all_token_ids


def sample_negative_examples(candidate_ids, pos_candidate_ids, per_query_nb_examples):
    """ Sample negative examples.

    Args:
    - candidate_ids: list of candidate IDs
    - pos_candidate_ids: list of lists of positive candidate IDs (one for each query)
    - per_query_nb_examples: sum of number of positive and negative examples per query
    
    """
    logger.info("  Sampling negative examples with per_query_nb_examples={}".format(per_query_nb_examples))
    # Sample a bunch of indices at once to save time on generating random candidate indices
    buffer_size = 1000000
    sampled_indices = np.random.randint(len(candidate_ids), size=buffer_size)
    neg_candidate_ids = []
    i = 0
    for pos in pos_candidate_ids:
        pos = set(pos)
        nb_neg = max(0, per_query_nb_examples-len(pos))
        neg = []
        while len(neg) < nb_neg:
            sampled_index = sampled_indices[i]
            i += 1 
            if i == buffer_size:
                # Sample more indices
                sampled_indices = np.random.randint(len(candidate_ids), size=buffer_size)
                i = 0
            if candidate_ids[sampled_index] not in pos:
                neg.append(candidate_ids[sampled_index])
        neg_candidate_ids.append(neg)
    return neg_candidate_ids


def make_dataset(tokenizer,
                 query_token_ids,
                 candidate_token_ids,
                 candidate_ids,
                 candidate_labels=None,
                 max_length=128,
                 pad_on_left=False,
                 pad_token=0,
                 pad_token_segment_id=0,
                 mask_padding_with_zero=True):
    """Create a dataset for hypernym discovery.

    Note: this code is based on transformers.glue_convert_examples_to_features.

    Args:
    - tokenizer
    - query_token_ids: list of lists containing the token IDs of a set of queries
    - candidate_token_ids: list of lists containing the token IDs of all candidate hypernyms
    - candidate_ids: list of lists containing the IDs of all the candidates to evaluate for a given query
    - candidate_labels: (optional) list of lists containing the labels of the candidate_ids (0 or 1)

    """
    assert len(query_token_ids) == len(candidate_ids)
    nb_queries = len(query_token_ids)
    nb_candidates = len(candidate_token_ids)
    nb_pos_examples = 0
    nb_neg_examples = 0
    if candidate_labels:
        for labels in candidate_labels:
            for label in labels:
                if label == 1:
                    nb_pos_examples += 1
                elif label == 0:
                    nb_neg_examples += 1
                else:
                    raise ValueError("unrecognized label '{}'".format(label))
    
    logger.info("***** Making dataset ******")
    logger.info("  Nb queries: {}".format(nb_queries))
    logger.info("  Nb candidates: {}".format(nb_candidates))
    if candidate_labels:
        logger.info("  Nb positive examples: {}".format(nb_pos_examples))
        logger.info("  Nb negative examples: {}".format(nb_neg_examples))
    logger.info("  Max length: {}".format(max_length))

    features = []
        
    # Loop over queries
    for i in range(len(query_token_ids)):
        q_tok_ids = query_token_ids[i]
        
        candidates = candidate_ids[i]
        labels = candidate_labels[i] if candidate_labels else [0] * len(candidates)

        # Loop over gold hypernyms of this query
        for candidate_id, label in zip(candidates, labels):
            g_tok_ids = candidate_token_ids[candidate_id]
            inputs = tokenizer.encode_plus(q_tok_ids,
                                           text_pair=g_tok_ids,
                                           add_special_tokens=True,
                                           max_length=max_length,
                                           pad_to_max_length=True)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
            

            features.append(InputFeatures(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          label=label))
        # Log some info on the last candidate for this query
        if i < 5:
            logger.info("*** Example ***")
            logger.info("  i: %d" % (i))
            logger.info("  query token IDs: {}".format(q_tok_ids))
            logger.info("  candidate token IDs: {}".format(g_tok_ids))
            logger.info("  input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("  attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("  token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("  label: %s" % (label))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)


def load_and_cache_dataset(args, tokenizer, set_type):
    """Load a dataset from source files or cache, and cache if
    necessary. 

    Dataset can be a training, dev or test set for hypernym discovery,
    or a list of candidate hypernyms. 

    Return a dict (whose keys depend on the set_type)

    """
    
    if set_type not in ["train", "dev", "test", "candidates"]:
        raise ValueError("unrecognized set_type '{}'".format(set_type))

    # Make sure only the first process in distributed training
    # processes the queries, and the others will use the cache. If
    # set_type is not 'train', then we assume we are in evaluation
    # mode.
    if args.local_rank not in [-1, 0] and set_type=='train':
        torch.distributed.barrier()  

    model_name = list(filter(None, args.model_name_or_path.split('/'))).pop()
    dir_cache = os.path.join(args.data_dir, "cache_for_{}_with_max_len_{}".format(model_name, args.max_seq_length))
    if not os.path.exists(dir_cache) and args.local_rank in [-1,0]:
        os.makedirs(dir_cache)

    # Check if cache file (pickled dict) exists for the given
    # set_type. If so, load data from cache. 
    path_cache = os.path.join(dir_cache, "{}.data.pkl".format(set_type))
    if os.path.exists(path_cache) and not args.overwrite_cache:
        logger.info("Loading {} data from pickle file {}".format(set_type, path_cache))
        with open(path_cache, 'rb') as reader:
            data = pickle.load(reader)
       
    else:
        # Load data from file, then cache (unless set_type is 'test' in which case caching is useless)
        data = {}

        # If we are processing the training, dev or test sets, make
        # sure we have the candidates first
        if set_type in ['train', 'dev', 'test']:
            candidate_data = load_and_cache_dataset(args, tokenizer, 'candidates')
            data["candidate_token_ids"] = candidate_data["candidate_token_ids"]
            data["candidate2id"] = candidate_data["candidate2id"] 
        logger.info("Loading {} data from source files in {}".format(set_type, args.data_dir))   
        if set_type == "candidates":
            # Load candidate2id (OrderedDicat that maps candidates, in
            # order in which they were read from source data, to their
            # index in that order (0-indexed), and candidate_token_ids
            # (list of lists, one per candidate, in same order as the
            # source file).
            path_candidates = os.path.join(args.data_dir, "candidates.txt")
            candidates = []
            with open(path_candidates) as f:
                for line in f:
                    candidates.append(line.strip())
            data["candidate2id"] = OrderedDict()
            for i,c in enumerate(candidates):
                data["candidate2id"][c] = i
            candidate_tokens, candidate_token_ids = tokenize_strings(tokenizer, candidates, max_length=args.MAX_CANDIDATE_LENGTH)
            data["candidate_token_ids"] = candidate_token_ids
            
        if set_type in ["train", "dev", "test"]:
            # Load query_token_ids (list of lists, one per query, in
            # same order as source file)
            path_queries = os.path.join(args.data_dir, "{}.queries.txt".format(set_type))
            queries = []
            with open(path_queries) as f:
                for line in f:
                    queries.append(line.strip())
            query_tokens, query_token_ids = tokenize_strings(tokenizer, queries, max_length=args.MAX_QUERY_LENGTH)
            data["queries"] = queries
            data["query_token_ids"] = query_token_ids

        path_gold_hypernyms = os.path.join(args.data_dir, '{}.gold.tsv'.format(set_type))            
        if (set_type in ["train", "dev"]) or (set_type=='test' and os.path.exists(path_gold_hypernyms)):
            # Load gold_hypernym_candidate_ids (list of lists, one per
            # query, same order as source file, IDs index the rows in
            # candidate_token_ids)
            gold_hypernyms = load_hypernyms(path_gold_hypernyms, normalize=False)
            gold_hypernym_candidate_ids = []
            for g_list in gold_hypernyms:
                g_id_list = []
                for g in g_list:
                    if g in data["candidate2id"]:
                        g_id_list.append(data["candidate2id"][g])
                    else:
                        raise KeyError("Gold hypernym '{}' not in candidate2id".format(g))
                gold_hypernym_candidate_ids.append(g_id_list)
            data["gold_hypernym_candidate_ids"] = gold_hypernym_candidate_ids

        if set_type=='train':
            # Load neg_candidate_ids (list of lists, one per query, in
            # same order as queries in source data, IDs index the rows
            # in candidate_token_ids) obtained by negative sampling
            candidate_ids = list(data["candidate2id"].values())
            neg_candidate_ids = sample_negative_examples(candidate_ids, data["gold_hypernym_candidate_ids"], args.per_query_nb_examples)
            data["neg_candidate_ids"] = neg_candidate_ids
            
        if set_type != 'test':
            # Cache data 
            if args.local_rank in [-1, 0]:
                logger.info("Saving pickled {} data in {}".format(set_type, path_cache))
                with open(path_cache, 'wb') as f:
                    pickle.dump(data, f)
            
    # Make sure only the first process in distributed training
    # processes the queries, and the others will use the cache. If
    # set_type is not 'train', then we assume we are in evaluation
    # mode.
    if args.local_rank == 0 and set_type=='train':
        torch.distributed.barrier()  

    # Return data
    return data

            
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
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

    parser.add_argument("--per_query_nb_examples", default=50, type=int, 
                        help=("Nb training examples per training query. "
                              "Nb negative examples is obtained by subtracting the number of positive examples for a given query."))
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
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
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
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
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Set up task
    task = "hyperdisco"
    num_labels = 2
    args.MAX_CANDIDATE_LENGTH = 20
    args.MAX_QUERY_LENGTH = 20
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=task,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Load candidates (which we need whether we are doing training or prediction)
    candidate_data = load_and_cache_dataset(args, tokenizer, 'candidates')
    candidate_token_ids = candidate_data["candidate_token_ids"]
    candidate2id = candidate_data["candidate2id"]

    # Training
    if args.do_train:
        train_data = load_and_cache_dataset(args, tokenizer, 'train')
        train_queries = train_data["queries"]
        train_query_token_ids = train_data["query_token_ids"]
        train_pos_candidate_ids = train_data["gold_hypernym_candidate_ids"]
        train_neg_candidate_ids = train_data["neg_candidate_ids"]
        candidate_ids = [x+y for (x,y) in zip(train_pos_candidate_ids, train_neg_candidate_ids)]
        candidate_labels = [[1]*len(x)+[0]*len(y) for (x,y) in zip(train_pos_candidate_ids, train_neg_candidate_ids)]
        train_dataset = make_dataset(tokenizer,
                                     train_query_token_ids,
                                     candidate_token_ids,
                                     candidate_ids,
                                     candidate_labels=candidate_labels,
                                     max_length=args.max_seq_length,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     mask_padding_with_zero=True)

        
        # Run training loop
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # Evaluation on dev set
    eval_results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer_to_eval = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints on dev set: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model_to_eval = model_class.from_pretrained(checkpoint)
            model_to_eval.to(args.device)
            result = evaluate(args, model_to_eval, tokenizer_to_eval, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            eval_results.update(result)

    # Prediction on test set
    if args.do_pred and args.local_rank in [-1, 0]:
        results = predict(args, model, tokenizer)
        eval_results.update(results)

        
    return eval_results


if __name__ == "__main__":
    results = main()
