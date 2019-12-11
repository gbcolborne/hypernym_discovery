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
from transformers.data.processors.utils import InputExample
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from sklearn.metrics import f1_score, average_precision_score

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


def train(args, train_dataset, model, tokenizer, label_list):
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
                        results = evaluate(args, model, tokenizer, label_list)
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
                    if not os.path.exists(output_dir):
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


def evaluate(args, model, tokenizer, label_list, prefix=""):
    """ Evaluate on dev set. """
    
    results = {}
    eval_examples, eval_dataset = load_and_cache_examples(args, tokenizer, label_list, 'dev')
    
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
        
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) # MUST BE SEQUENTIAL!
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
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
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      inputs['labels'].detach().cpu().numpy(),
                                      axis=0)
    eval_loss = eval_loss / nb_eval_steps

    # This function assumes the rows in preds are in the same order as
    # the examples, which is why we had to use a sequential data
    # loader.
    q2data = map_queries_to_pred(eval_examples, preds) 

    # Compute evaluation metrics
    result = {}
    # Evaluate classification accuracy
    y_pred = np.argmax(preds, axis=1)
    result["acc"] = (y_pred==out_label_ids).mean()
    result["f1"] = f1_score(y_true=out_label_ids, y_pred=y_pred)
    # Evaluate per-query ranking of candidates (which are limited to
    # the positive examples and a small sample of negative examples --
    # evaluating on all candidates at each validation step would be
    # much more expensive)
    k = 15
    ap_values = []
    for i,q in enumerate(q2data.keys()):
        topk = sorted(q2data[q], key=lambda x:x[1], reverse=True)[:k]
        y_true = [yt for (c,p,yp,yt) in topk]
        y_score = [p for (c,p,yp,yt) in topk]
        ap = average_precision_score(y_true=y_true, y_score=y_score)
        ap_values.append(ap)
    result["MAP"] = np.mean(ap_values)
    results.update(result)
    
    output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            
    return results


def predict(args, model, tokenizer, label_list):
    """ Run prediction on test set. """

    # Get test data
    test_examples, eval_dataset = load_and_cache_examples(args, tokenizer, label_list, 'test')

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) # MUST BE SEQUENTIAL!
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Run prediction
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Predicting"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            _, logits = outputs[:2]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    # This function assumes the rows in preds are in the same order as
    # the examples, which is why we had to use a sequential data
    # loader.
    q2data = map_queries_to_pred(test_examples, preds) 
            
    # Write probabilities for all candidates for each test query.
    path = os.path.join(args.output_dir, "test_probs.tsv")
    logger.info("Writing candidate probabilities for each query to {}".format(path))
    with open(path, 'w') as f:
        for q in q2data.keys():
            for (c, p, yp, yt) in q2data[q]:
                f.write("{}\t{}\t{}\n".format(q, c, p))
                
    # Write and log top k candidates for each test query. If we have
    # the gold labels, evaluate ranking of candidates wrt gold
    # hypernyms.
    k = 15
    path_c = os.path.join(args.output_dir, "test_top15_candidates.tsv")
    path_p = os.path.join(args.output_dir, "test_top15_probabilities.tsv")
    logger.info("***** Ranking candidates *****")
    logger.info("  Writing top {} candidates for each query to {}".format(k, path_c))
    logger.info("  Writing top {} probabilities for each query to {}".format(k, path_p))
    if gold_available:
        logger.info("  Evaluating ranking of candidates wrt gold hypernyms in {}".format(path_gold))
        ap_values = []
    with open(path_c, 'w') as fc, open(path_p, 'w') as fp:
        for i,q in enumerate(q2data.keys()):
            topk = sorted(q2data[q], key=lambda x:x[1], reverse=True)[:k]
            topk_string = ', '.join(["('{}',{:.5f})".format(c,p) for (c,p,yp,yt) in topk])
            logger.info("{}. Top candidates for '{}': {}".format(i+1, q, topk_string))
            fc.write("{}\n".format("\t".join([c for (c,p,yp,yt) in topk])))
            fp.write("{:.5f}\n".format("\t".join([p for (c,p,yp,yt) in topk])))
            if gold_available:
                y_true = [yt for (c,p,yp,yt) in topk]
                y_score = [p for (c,p,yp,yt) in topk]
                ap = average_precision_score(y_true=y_true, y_score=y_score)
                if np.isnan:
                    logger.warning("NaN resulted from computing AP(y_true={},y_score={})".format(y_true, y_score))
                    sys.exit()
                ap_values.append(ap)

    # If gold labels are not available, we are done
    if not gold_available:
        return None

    # Write average precision of each query
    output_eval_file = os.path.join(args.output_dir, "test_average_precision.txt")
    with open(output_eval_file, "w") as writer:
        for ap in ap_values:
            writer.write("{:.5f}\n".format(ap))

    # Return evaluation results. To have a similar output as the
    # evaluate function, we will return a dict of results containing a
    # single dict with all our results.
    results = {}
    result = {}
    result["ap"] = ap_values
    result["map"] = np.mean(ap_values)
    results.update(result)
    logger.info("***** Eval results *****")
    logger.info("  MAP = %s", str(result["map"]))

    return results


def map_queries_to_pred(examples, probs):
    """Given a list of n examples, and a nX2 matrix of class
    probabilities, create an OrderedDict that maps queries to a list
    of (candidate, probability, predicted class, true class) tuples.
    Note: we assume the rows in probs are in the same order as the
    examples.

    """
    logger.info("Mapping queries to candidate probabilities")
    pred_class = np.argmax(probs, axis=1)
    pred_prob = probs[:,1]
    q2data = OrderedDict()
    for i in range(len(examples)):
        q = examples[i].text_a
        c = examples[i].text_b
        ytrue = examples[i].label
        ypred = pred_class[i]
        prob = pred_prob[i]
        if q not in q2data:
            q2data[q] = []
        q2data[q].append((c, prob, ypred, ytrue))
    return q2data


def create_examples(args, path_queries, path_candidates, set_type, path_gold=None):
    if set_type not in ["train", "dev", "test"]:
        raise ValueError("unrecognized set_type '{}'".format(set_type))
    if path_gold is None and set_type != 'test':
        raise ValueError("path_gold must be provided")

    # Load candidates
    with open(path_candidates) as f:
        candidates = []
        for line in f:
            candidate = line.strip()
            candidates.append(candidate)
    nb_candidates = len(candidates)
    logger.info("  Nb candidates: {}".format(nb_candidates))

    # Load queries
    queries = []
    with open(path_queries) as fq:
        for line in fq:
            query = line.strip()
            queries.append(query)
    logger.info("  Nb queries: {}".format(len(set(queries))))
    
    # Load gold hypernyms
    pos = {q:[] for q in set(queries)}
    if path_gold:
        with open(path_gold) as fg:
            for i,line in enumerate(fg):
                gold = line.strip()
                query = queries[i]
                pos[query].append(gold)
        logger.info("  Nb positive examples: {}".format(sum(len(v) for k,v in pos.items())))

    # Identify or sample negative examples
    neg = {}
    if set_type == 'test':
        for q in pos:
            if path_gold is None:
                neg[q] = candidates[:]
            else:
                neg[q] = list(filter(lambda x:x not in pos[q], candidates))
                logger.info("  Nb negative examples for query '{}': {}".format(q, len(neg[q])))
                
    else:
        # Sample a bunch of indices at once to save time on generating random candidate indices
        logger.info("  Sampling negative examples with per_query_nb_examples={}".format(args.per_query_nb_examples))
        buffer_size = 1000000
        sampled_indices = np.random.randint(nb_candidates, size=buffer_size)
        i = 0
        for q in pos:
            neg[q] = []
            nb_neg_examples = max(0, args.per_query_nb_examples-len(pos[q]))
            nb_added = 0
            while nb_added < nb_neg_examples:
                sampled_index = sampled_indices[i]
                i += 1 
                if i == buffer_size:
                    # Sample more indices
                    sampled_indices = np.random.randint(nb_candidates, size=buffer_size)
                    i = 0
                if candidates[sampled_index] not in pos[q]:
                    neg[q].append(candidates[sampled_index])
                    nb_added += 1
        nb_pos = sum(len(v) for k,v in pos.items())
        nb_neg = sum(len(v) for k,v in neg.items())
        logger.info("  Nb positive examples: {}".format(nb_pos))
        logger.info("  Nb negative examples: {}".format(nb_neg))
        
    # Create input examples
    examples = []
    q_id = 0
    for q in pos:
        q_id += 1
        nb_examples_for_q = 0
        for (label, hlist) in [(1,pos[q]),(0,neg[q])]:
            for h in hlist:
                guid = "%s-%s" % (set_type, len(examples)+1)
                examples.append(InputExample(guid=guid, text_a=q, text_b=h, label=label))
                nb_examples_for_q += 1
        logger.info("  Created {} examples for query ' ' ({}/{})".format(nb_examples_for_q, q, q_id, len(pos))) 
    return examples

def get_train_examples(args):
    path_queries = os.path.join(args.data_dir, "train.queries.txt")
    path_gold = os.path.join(args.data_dir, "train.gold.txt")
    path_candidates = os.path.join(args.data_dir, "candidates.txt")
    logger.info("LOOKING AT {}".format(args.data_dir))
    return create_examples(args, path_queries, path_candidates, "train", path_gold=path_gold)

def get_dev_examples(args):
    path_queries = os.path.join(args.data_dir, "dev.queries.txt")
    path_gold = os.path.join(args.data_dir, "dev.gold.txt")
    path_candidates = os.path.join(args.data_dir, "candidates.txt")
    return create_examples(args, path_queries, path_candidates, "dev", path_gold=path_gold)

def get_test_examples(args, gold_available=False):
    path_queries = os.path.join(args.data_dir, "test.queries.txt")
    if gold_available:
        path_gold = os.path.join(args.data_dir, "test.gold.txt")
    else:
        path_gold = None
    path_candidates = os.path.join(args.data_dir, "candidates.txt")
    return create_examples(args, path_queries, path_candidates, "test", path_gold=path_gold)


def load_and_cache_examples(args, tokenizer, label_list, set_type):
    """Load and cache examples and features."""
    if set_type not in ["train", "dev", "test"]:
        raise ValueError("Unrecognized set_type '{}'".format(set_type))
    if args.local_rank not in [-1, 0] and set_type=='train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Make file name for cached examples and features
    suffix =  '{}_{}_{}_{}'.format(
        set_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        args.per_query_nb_examples)
    cached_examples_file = os.path.join(args.data_dir, 'cached_examples_{}'.format(suffix))
    cached_features_file = os.path.join(args.data_dir, 'cached_features_{}'.format(suffix))

    # Load examples from cache or dataset file
    if os.path.exists(cached_examples_file) and not args.overwrite_cache:
        logger.info("Loading examples from cached file %s", cached_examples_file)
        with open(cached_examples_file, 'rb') as f:
            examples = pickle.load(f)
    else:
        logger.info("Creating examples from %s set at %s", set_type, args.data_dir)
        if set_type=='train':
            examples = get_train_examples(args)
        elif set_type=='dev':
            examples = get_dev_examples(args)
        elif set_type=='test':
            # Check if gold labels are available
            path_gold = os.path.join(args.data_dir, "test.gold.txt")
            gold_available = os.path.exists(path_gold)
            examples = get_test_examples(args, gold_available=gold_available)
        if args.local_rank in [-1, 0]:
            logger.info("Saving examples into cached file %s", cached_examples_file)
            with open(cached_examples_file, 'wb') as f:
                pickle.dump(examples, f)

    # Load features from cache or dataset file        
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from %s set at %s", set_type, args.data_dir)        
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=args.output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    features = convert_features_to_tensor(features)
    
    if args.local_rank == 0 and set_type=='train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return examples, features

def convert_features_to_tensor(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

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

    # Prepare task
    args.output_mode = "classification"
    label_list = [0,1]
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task="HyperDisco",
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


    # Training
    if args.do_train:
        _, train_dataset = load_and_cache_examples(args, tokenizer, label_list, 'train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, label_list)
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
            result = evaluate(args, model_to_eval, tokenizer_to_eval, label_list, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            eval_results.update(result)

    # Prediction on test set
    if args.do_pred and args.local_rank in [-1, 0]:
        _ = predict(args, model, tokenizer, label_list)

        
    return eval_results


if __name__ == "__main__":
    main()
