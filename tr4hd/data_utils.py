import glob, os, re, logging, shutil
import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def make_candidate_set(opt, tokenizer, candidate_data):
    """ Make unlabeled dataset for candidates.
    Args:
    - opt:
    - tokenizer:
    - candidate_data: dict containing candidates (list)

    """
    candidates = candidate_data['candidates']
    return make_q_or_c_dataset(opt,
                               tokenizer, 
                               candidates,
                               max_length=128, 
                               pad_on_left=False, 
                               pad_token=0, 
                               pad_token_segment_id=0, 
                               mask_padding_with_zero=True)

def make_test_set(opt, tokenizer, test_data):
    """ Make unlabeled dataset for test set.
    Args:
    - opt:
    - tokenizer:
    - test_data: dict containing queries (list)

    """
    queries = test_data["queries"]
    return make_q_or_c_dataset(opt,
                               tokenizer, 
                               queries,
                               max_length=128, 
                               pad_on_left=False, 
                               pad_token=0, 
                               pad_token_segment_id=0, 
                               mask_padding_with_zero=True)

def make_train_set(opt, tokenizer, train_data):
    """ Make labeled dataset for training set. Subsample candidates using negative sampling.
    Args:
    - opt:
    - tokenizer: 
    - train_data: dict containing queries (list), candidates (list), and gold_hypernym_candidate_ids (list of lists, one per query)

    """

    # Load training data
    queries = train_data["queries"]
    candidates = train_data["candidates"]
    gold_cand_ids = train_data["gold_hypernym_candidate_ids"]

    # Subsample candidates for training (i.e. negative sampling).
    all_cand_ids = list(range(len(candidates)))
    neg_cand_ids = sample_negative_examples(all_cand_ids, gold_cand_ids, opt.per_query_nb_examples)
    cand_ids = []
    labels = []
    for i in range(len(queries)):
        xy = [(x,1) for x in gold_cand_ids[i]] + [(x,0) for x in neg_cand_ids[i]]
        np.random.shuffle(xy)
        x,y = zip(*xy)
        cand_ids.append(x)
        labels.append(y)

    # Build dataset
    return make_q_and_c_dataset(opt,
                                tokenizer,
                                queries,
                                cand_ids,
                                candidate_labels=labels,
                                max_length=128,
                                pad_on_left=False,
                                pad_token=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True,
                                verbose=True)

def make_dev_set(opt, tokenizer, dev_data):
    """ Make labeled dataset for validation data. Include all candidates for evaluation.
    Args:
    - opt:
    - tokenizer:
    - dev_data: dict containing queries (list), candidates (list), and gold_hypernym_candidate_ids (list of lists, one per query).

    """

    # Load validation data
    queries = dev_data["queries"]
    gold_cand_ids = dev_data["gold_hypernym_candidate_ids"]
    nb_candidates = len(dev_data["candidates"])
    all_cand_ids = list(range(nb_candidates))
    cand_ids = []
    labels = []
    for i in range(len(queries)):
        y = [0] * nb_candidates
        for c in gold_cand_ids[i]:
            y[c] = 1
        cand_ids.append(all_cand_ids[:])
        labels.append(y)

    # Build dataset
    return make_q_and_c_dataset(opt,
                                tokenizer,
                                queries,
                                cand_ids,
                                candidate_labels=labels,
                                max_length=128,
                                pad_on_left=False,
                                pad_token=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True)


def sample_negative_examples(candidate_ids, pos_candidate_ids, per_query_nb_examples):
    """ Sample negative examples.

    Args:
    - candidate_ids: list of candidate IDs
    - pos_candidate_ids: list of lists of positive candidate IDs (one for each query)
    - per_query_nb_examples: sum of number of positive and negative examples per query. Note: if any queries have more than this number of positive examples, some will be discarded.
    
    """
    
    logger.info("  Sampling negative examples with per_query_nb_examples={}".format(per_query_nb_examples))
    # Sample a bunch of indices at once to save time on generating random candidate indices
    buffer_size = 1000000
    sampled_indices = np.random.randint(len(candidate_ids), size=buffer_size)
    nb_queries = len(pos_candidate_ids)
    neg_candidate_ids = []
    nb_pos_discarded = 0
    buffer_index = 0
    for i in range(nb_queries):
        pos = pos_candidate_ids[i]
        if len(pos) > per_query_nb_examples:
            nb_pos_discarded += len(pos) - per_query_nb_examples
            pos_candidate_ids[i] = pos[:per_query_nb_examples]
        pos_set = set(pos)
        nb_neg = max(0, per_query_nb_examples-len(pos))
        neg = []
        while len(neg) < nb_neg:
            sampled_index = sampled_indices[buffer_index]
            buffer_index += 1 
            if buffer_index == buffer_size:
                # Sample more indices
                sampled_indices = np.random.randint(len(candidate_ids), size=buffer_size)
                buffer_index = 0
            if candidate_ids[sampled_index] not in pos_set:
                neg.append(candidate_ids[sampled_index])
        neg_candidate_ids.append(neg)
    if nb_pos_discarded > 0:
        msg = "  {} positive hypernyms removed because the query had more than {}".format(nb_pos_discarded, per_query_nb_examples)
        logger.warning(msg)
    return neg_candidate_ids


def load_hypernyms(path):
    """Given the path of a hypernyms file, return list of lists of
    hypernyms.

    """
    with open(path) as f:
        hypernyms = []
        for line in f:
            h_list = line.strip().split("\t")
            hypernyms.append(h_list)
    return hypernyms

def encode_token_ids(opt,
                     token_ids, 
                     tokenizer, 
                     max_length=128, 
                     pad_on_left=False, 
                     pad_token=0, 
                     pad_token_segment_id=0, 
                     mask_padding_with_zero=True):
    """ Given a list of token_ids, encode the token_ids, adding special tokens and padding. 
    
    """
    inputs = tokenizer.encode_plus(token_ids,
                                   add_special_tokens=True,
                                   max_length=max_length,
                                   pad_to_max_length=True)
    input_ids, tok_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    att_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Encode language
    if opt.encoder_type == 'xlm':
        lang_id = tokenizer.lang2id[opt.lang]
        langs = [lang_id] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        att_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + att_mask
        tok_type_ids = ([pad_token_segment_id] * padding_length) + tok_type_ids
        if opt.encoder_type == 'xlm':
            langs = ([pad_token] * padding_length) + langs
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        att_mask = att_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        tok_type_ids = tok_type_ids + ([pad_token_segment_id] * padding_length)
        if opt.encoder_type == 'xlm':
            langs = langs + ([pad_token] * padding_length)
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(att_mask) == max_length, "Error with input length {} vs {}".format(len(att_mask), max_length)
    assert len(tok_type_ids) == max_length, "Error with input length {} vs {}".format(len(tok_type_ids), max_length)
    if opt.encoder_type == 'xlm':
        assert len(langs) == max_length, "Error with input length {} vs {}".format(len(langs), max_length)
    inputs = {"input_ids":input_ids, "token_type_ids":tok_type_ids, "attention_mask":att_mask}
    if opt.encoder_type == 'xlm':
        inputs['langs'] = langs
    return inputs


def encode_string_inputs(opt,
                         tokenizer, 
                         strings,
                         max_length=128,
                         pad_on_left=False,
                         pad_token=0,
                         pad_token_segment_id=0,
                         mask_padding_with_zero=True):
    """ Tokenize strings and create dict containing: input_ids, attention_mask, token_type_ids, langs. 

    """
    all_tok_ids = []
    all_att_mask = []
    all_tok_type_ids = []
    if opt.encoder_type == 'xlm':
        all_langs = []
    nb_processed = 0
    for string in strings:
        toks = tokenizer.tokenize(string)
        tok_ids = tokenizer.encode(toks, add_special_tokens=False, max_length=max_length, pad_to_max_length=False)
        inputs = encode_token_ids(opt,
                                  tok_ids, 
                                  tokenizer, 
                                  max_length=max_length, 
                                  pad_on_left=pad_on_left,
                                  pad_token=pad_token, 
                                  pad_token_segment_id=pad_token_segment_id, 
                                  mask_padding_with_zero=mask_padding_with_zero)
        all_tok_ids.append(inputs["input_ids"])
        all_att_mask.append(inputs["attention_mask"])
        all_tok_type_ids.append(inputs["token_type_ids"])
        if opt.encoder_type == 'xlm':
            all_langs.append(inputs['langs'])
        nb_processed += 1
        if nb_processed % 1000 == 0:
            logger.info("  Nb strings processed: {}".format(nb_processed))
    inputs = {}
    inputs["input_ids"] = all_tok_ids
    inputs["attention_mask"] = all_att_mask
    inputs["token_type_ids"] = all_tok_type_ids
    if opt.encoder_type == 'xlm':
        inputs["langs"] = all_langs
    return inputs

def make_q_or_c_dataset(opt,
                        tokenizer, 
                        strings,
                        max_length=128, 
                        pad_on_left=False, 
                        pad_token=0, 
                        pad_token_segment_id=0, 
                        mask_padding_with_zero=True):
    """ Create an unlabeled dataset for inputs to an encoder (for queries or candidates). 

    """
    nb_strings = len(strings)
    logger.info("***** Making dataset ******")
    logger.info("  Nb strings (queries or candidates): {}".format(nb_strings))
    logger.info("  Max length: {}".format(max_length))
    inputs = encode_string_inputs(opt,
                                  tokenizer, 
                                  strings,
                                  max_length=max_length, 
                                  pad_on_left=pad_on_left, 
                                  pad_token=pad_token, 
                                  pad_token_segment_id=pad_token_segment_id, 
                                  mask_padding_with_zero=mask_padding_with_zero)
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
    if opt.encoder_type == 'xlm':
        langs = torch.tensor(inputs['langs'], dtype=torch.long)
        return TensorDataset(input_ids, attention_mask, token_type_ids, langs)
    else:
        return TensorDataset(input_ids, attention_mask, token_type_ids)



def make_q_and_c_dataset(opt,
                         tokenizer,
                         queries,
                         candidate_ids,
                         candidate_labels=None,
                         max_length=128,
                         pad_on_left=False,
                         pad_token=0,
                         pad_token_segment_id=0,
                         mask_padding_with_zero=True,
                         verbose=False):
    """Create a dataset for query inputs and candidate ids, with optional labels.

    Args:
    - opt
    - tokenizer
    - queries: list of query strings
    - candidate_ids: list of lists containing the IDs of all the candidates to evaluate for a given query
    - candidate_labels: (optional) list of lists containing the labels of the candidate_ids (0 or 1)

    """

    # Check args
    assert len(queries) == len(candidate_ids)
    nb_candidates_fd = {}
    for c in candidate_ids:
        length = len(c)
        if length not in nb_candidates_fd:
            nb_candidates_fd[length] = 0
        nb_candidates_fd[length] += 1
    if len(nb_candidates_fd) > 1:
        msg = "Nb candidates must be same for all queries. "
        msg += "Found the following numbers: %s" % ", ".join(["{} (count={})".format(k,v) for (k,v) in nb_candidates_fd.items()])
        raise ValueError(msg)

    nb_queries = len(queries)
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
    if candidate_labels:
        logger.info("  Nb positive examples: {}".format(nb_pos_examples))
        logger.info("  Nb negative examples: {}".format(nb_neg_examples))
    logger.info("  Max length: {}".format(max_length))

    inputs = encode_string_inputs(opt,
                                  tokenizer, 
                                  queries,
                                  max_length=max_length,
                                  pad_on_left=pad_on_left,
                                  pad_token=pad_token,
                                  pad_token_segment_id=pad_token_segment_id,
                                  mask_padding_with_zero=mask_padding_with_zero)
    q_tok_ids = inputs["input_ids"]
    q_att_mask = inputs["attention_mask"]
    q_tok_type_ids = inputs["token_type_ids"]
    if opt.encoder_type == 'xlm':
        q_langs = inputs['langs']
    if candidate_labels is None:
        [[0] * len(candidates)] * len(queries)

    # Log a few examples
    if verbose:
        for i in range(5):
            logger.info("*** Example ***")
            logger.info("  i: %d" % (i))
            logger.info("  query: %s" % queries[i])
            logger.info("  query token IDs: {}".format(q_tok_ids[i]))
            logger.info("  attention_mask: %s" % " ".join([str(x) for x in q_att_mask[i]]))
            logger.info("  token type ids: %s" % " ".join([str(x) for x in q_tok_type_ids[i]]))
            if opt.encoder_type == 'xlm':
                logger.info("  langs: %s" % " ".join([str(x) for x in q_langs[i]]))
            logger.info("  candidate ids: %s" % " ".join([str(x) for x in candidate_ids[i]]))
            logger.info("  candidate labels: %s" % " ".join([str(x) for x in candidate_labels[i]]))

    q_tok_ids = torch.tensor(q_tok_ids, dtype=torch.long)
    q_att_mask = torch.tensor(q_att_mask, dtype=torch.long)
    q_tok_type_ids = torch.tensor(q_tok_type_ids, dtype=torch.long)
    candidate_ids = torch.tensor(candidate_ids, dtype=torch.long)
    candidate_labels = torch.tensor(candidate_labels, dtype=torch.long)
    if opt.encoder_type == 'xlm':
        q_langs = torch.tensor(q_langs, dtype=torch.long)
        return TensorDataset(q_tok_ids, q_att_mask, q_tok_type_ids, q_langs, candidate_ids, candidate_labels)
    else:
        return TensorDataset(q_tok_ids, q_att_mask, q_tok_type_ids, candidate_ids, candidate_labels)


def load_hd_data(opt, set_type):
    """Load data from file. 
    Dataset can be a training, dev or test set for hypernym discovery,
    or a list of candidate hypernyms. 
    Return a dict containing: candidates and candidate2id, a list of queries (if the set type is not candidates) and a list of lists of gold hypernym candidate IDs (if the set type is train or dev).
    """
    
    if set_type not in ["train", "dev", "test", "candidates"]:
        raise ValueError("unrecognized set_type '{}'".format(set_type))

    # Load candidates, which we need regardless of the set type
    path_candidates = os.path.join(opt.data_dir, "candidates.txt")
    candidates = []
    with open(path_candidates) as f:
        for line in f:
            candidates.append(line.strip())
    data = {}
    data["candidates"] = candidates
    data["candidate2id"] = {x:i for (i,x) in enumerate(candidates)}
    if set_type == "candidates":
        return data

    # Load queries
    path_queries = os.path.join(opt.data_dir, "{}.queries.txt".format(set_type))
    queries = []
    with open(path_queries) as f:
        for line in f:
            queries.append(line.strip())
    data["queries"] = queries

    # Load gold_hypernym_candidate_ids (list of lists, one per
    # query, same order as source file)
    path_gold_hypernyms = os.path.join(opt.data_dir, '{}.gold.tsv'.format(set_type))            
    if (set_type in ["train", "dev"]) or (set_type=='test' and os.path.exists(path_gold_hypernyms)):
        gold_hypernyms = load_hypernyms(path_gold_hypernyms)
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
    return data

            
def rotate_checkpoints(save_total_limit, output_dir, checkpoint_prefix, use_mtime=False):
    if not save_total_limit:
        return
    if save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= save_total_limit:
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
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
