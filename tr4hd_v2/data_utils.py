import glob, os, re, logging, shutil, math, random
import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

PAD_TOKEN=0
SEGMENT_ID=0
MASK_PADDING_WITH_ZERO=True

def make_train_set(opt, tokenizer, train_data, seed=0, verbose=False):
    """ Make labeled dataset for training set. Subsample candidates using negative sampling.
    Args:
    - opt:
    - tokenizer: 
    - train_data: dict containing queries (list), candidates (list), and gold_hypernym_candidate_ids (list of lists, one per query)

    """
    if opt.nb_neg_samples < 1:
        msg = "nb_neg_samples must be strictly positive"
        raise ValueError(msg)

    # Seed RNGs
    random.seed(seed)
    np.random.seed(seed)
    
    # Load training data
    queries = train_data["queries"]
    gold_cand_ids = train_data["gold_hypernym_candidate_ids"]
    candidates = train_data["candidates"]
    q2gold = {}

    # Map queries to gold hypernyms
    for i in range(len(queries)):
        q2gold[queries[i]] = set(gold_cand_ids[i])
    
    # Compute hypernym frequencies
    hyp_fd = {}
    for sub in gold_cand_ids:
        for hyp in sub:
            if hyp not in hyp_fd:
                hyp_fd[hyp] = 0
            hyp_fd[hyp] += 1
    min_freq = min(hyp_fd.values())

    # Compute negative sampling probabilities (with Laplace smoothing)
    if opt.smoothe_neg_sampling:
        counts = np.ones(len(candidates), dtype=float)
    else:
        counts = np.zeros(len(candidates), dtype=float)
    for c,f in hyp_fd.items():
        counts[c] += f
    neg_sampling_probs = counts / counts.sum()

    # Subsample positive examples based on gold hypernym frequencies
    if opt.pos_subsampling_factor > 0.0:
        # Compute sampling probabilities
        sample_probs = {}
        for hyp, freq in hyp_fd.items():
            sample_probs[hyp] = opt.pos_subsampling_factor / (freq - min_freq + 1) + 1 - opt.pos_subsampling_factor
        # Apply subsampling
        q_tmp = []
        h_tmp = []
        for query, hyps in zip(queries, gold_cand_ids):
            sub = []
            for hyp in hyps:
                if random.random() < sample_probs[hyp]:
                    sub.append(hyp)
            if len(sub):
                q_tmp.append(query)
                h_tmp.append(sub)
        queries = q_tmp
        gold_cand_ids = h_tmp
        
    # Log some stats
    nb_candidates = len(candidates)
    nb_queries = len(queries)
    nb_pos_examples = sum(len(x) for x in gold_cand_ids)
    nb_neg_samples = nb_pos_examples * opt.nb_neg_samples
    if verbose:
        logger.info("***** Making training set ******")
        logger.info("  Nb queries: {}".format(nb_queries))
        logger.info("  Max length: {}".format(opt.max_seq_length))
        logger.info("  Positive example subsampling factor: {}".format(opt.pos_subsampling_factor))        
        logger.info("  Nb positive examples kept: {}".format(nb_pos_examples))
        logger.info("  Nb negative examples sampled: {}".format(nb_neg_samples))
        logger.info("  Min negative sampling probability: {}".format(neg_sampling_probs.min()))
        logger.info("  Max negative sampling probability: {}".format(neg_sampling_probs.max()))        
        
    # Sample a bunch of hypernyms which we will use as negative examples
    buffer_size = 1000000
    cand_ids = np.arange(len(candidates))
    sampled_cand_ids = np.random.choice(cand_ids, size=buffer_size, replace=True, p=neg_sampling_probs)
    buffer_index = 0

    # Negative sampling
    queries_dup = []
    cand_ids = []
    labels = []
    order = list(range(opt.nb_neg_samples + 1))
    for i in range(nb_queries):
        for j in range(len(gold_cand_ids[i])):
            queries_dup.append(queries[i])
            cand_id_sample = []
            # Sample negative examples
            while len(cand_id_sample) < opt.nb_neg_samples:
                cand_id = sampled_cand_ids[buffer_index]
                if cand_id not in q2gold[queries[i]]:
                    cand_id_sample.append(cand_id)
                buffer_index += 1
                # Check if we should refresh buffer
                if buffer_index == buffer_size:
                    sampled_cand_ids = np.random.choice(cand_ids, size=buffer_size, replace=True, p=neg_sampling_probs)
                    buffer_index = 0
            cand_id_sample.append(gold_cand_ids[i][j])                    
            # Shuffle in a way where we remember where our target goes
            np.random.shuffle(order)
            shuffled = [0] * (opt.nb_neg_samples + 1)
            for ordered_index, random_index in enumerate(order):
                shuffled[random_index] = cand_id_sample[ordered_index]
            cand_ids.append(shuffled)
            target_ix = order[-1]
            labels.append(target_ix)

    # Shuffle
    order = list(range(len(queries_dup)))
    np.random.shuffle(order)
    queries_dup = [queries_dup[i] for i in order]
    cand_ids = [cand_ids[i] for i in order]
    labels = [labels[i] for i in order]
    
    # Encode queries
    query_input_ids, query_nb_tokens = encode_string_inputs(opt, tokenizer, queries_dup, verbose=verbose)

    # Log a few examples
    if verbose:
        for i in range(5):
            logger.info("*** Example ***")
            logger.info("  i: %d" % (i))
            logger.info("  query: %s" % queries_dup[i])
            logger.info("  query token IDs: {}".format(query_input_ids[i]))
            logger.info("  nb tokens (without padding): {}".format(query_nb_tokens[i]))
            logger.info("  candidate ids: %s" % " ".join([str(x) for x in cand_ids[i]]))
            logger.info("  label: %d" % labels[i])

    cand_ids = torch.tensor(cand_ids, dtype=torch.long, requires_grad=False, device=opt.device)
    labels = torch.tensor(labels, dtype=torch.long, requires_grad=False, device=opt.device)
    dataset = TensorDataset(query_input_ids, query_nb_tokens, cand_ids, labels)
    return dataset


def make_dev_set(opt, tokenizer, dev_data, verbose=False):
    """ Make labeled dataset for validation data. Include all candidates for evaluation.
    Args:
    - opt:
    - tokenizer:
    - dev_data: dict containing queries (list), candidates (list), and gold_hypernym_candidate_ids (list of lists, one per query).

    """

    # Load validation data
    queries = dev_data["queries"]
    pos_cand_ids = dev_data["gold_hypernym_candidate_ids"]
    nb_candidates = len(dev_data["candidates"])
    nb_pos_examples = sum(len(x) for x in pos_cand_ids)
    
    if verbose:
        logger.info("***** Making dev set ******")
        logger.info("  Nb queries: {}".format(len(queries)))
        logger.info("  Nb candidates: {}".format(nb_candidates))
        logger.info("  Nb positive examples: {}".format(nb_pos_examples))
        logger.info("  Max length: {}".format(opt.max_seq_length))

    # Pad gold candidate IDs and store number of gold cand IDs (not including padding)
    nb_pos_cand_ids = [len(x) for x in pos_cand_ids]
    max_pos = max(nb_pos_cand_ids)
    padding_id = -1
    padded_pos_cand_ids = []
    for pos in pos_cand_ids:
        padding = [padding_id] * (max_pos - len(pos))
        padded_pos_cand_ids.append(pos + padding)

    # Encode queries
    query_input_ids, query_nb_tokens = encode_string_inputs(opt, tokenizer, queries, verbose=verbose)

    # Log a few examples
    if verbose:
        for i in range(5):
            logger.info("*** Example *** ")
            logger.info("  i: %d" % (i))
            logger.info("  query: %s" % queries[i])
            logger.info("  query token IDs: {}".format(query_input_ids[i]))
            logger.info("  nb tokens (without padding): {}".format(query_nb_tokens[i]))
            logger.info("  gold candidate ids: %s" % " ".join([str(x) for x in pos_cand_ids[i]]))
            logger.info("  padded gold candidate ids: %s" % " ".join([str(x) for x in padded_pos_cand_ids[i]]))
            logger.info("  nb gold candidate ids (w/o padding): %d" % nb_pos_cand_ids[i])

    padded_pos_cand_ids = torch.tensor(padded_pos_cand_ids, dtype=torch.long, requires_grad=False, device=opt.device)
    nb_pos_cand_ids = torch.tensor(nb_pos_cand_ids, dtype=torch.long, requires_grad=False, device=opt.device)
    dataset = TensorDataset(query_input_ids, query_nb_tokens, padded_pos_cand_ids, nb_pos_cand_ids)
    return dataset


def make_test_set(opt, tokenizer, test_data, verbose=False):
    """ Make unlabeled dataset for test set.
    Args:
    - opt:
    - tokenizer:
    - test_data: dict containing queries (list)

    """
    queries = test_data["queries"]
    if verbose:
        logger.info("***** Making dataset of test query inputs ******")
        logger.info("  Nb queries: {}".format(len(queries)))
        logger.info("  Max length: {}".format(opt.max_seq_length))
    input_ids, nb_tokens = encode_string_inputs(opt, tokenizer, queries, verbose=verbose)
    return TensorDataset(input_ids, nb_tokens)


def make_candidate_set(opt, tokenizer, candidate_data, verbose=False):
    """ Make unlabeled dataset for candidates.
    Args:
    - opt:
    - tokenizer:
    - candidate_data: dict containing candidates (list)

    """
    candidates = candidate_data['candidates']
    if verbose:
        logger.info("***** Making dataset of candidate inputs ******")
        logger.info("  Nb candidates: {}".format(len(candidates)))
        logger.info("  Max length: {}".format(opt.max_seq_length))
    input_ids, nb_tokens = encode_string_inputs(opt, tokenizer, candidates, verbose=verbose)
    return TensorDataset(input_ids, nb_tokens)


def encode_string_inputs(opt, tokenizer, strings, verbose=False):
    """ Tokenize strings and return 2 tensors: input_ids (padded), nb_tokens (not including padding)

    """
    input_ids = []
    nb_tokens = []
    nb_processed = 0
    for string in strings:
        tokens = tokenizer.tokenize(string)
        token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length=opt.max_seq_length, pad_to_max_length=False)
        nb_tokens.append([len(token_ids)])
        # Pad
        padding_length = opt.max_seq_length - len(token_ids)
        token_ids += [PAD_TOKEN] * padding_length
        input_ids.append(token_ids)
        nb_processed += 1
        if verbose and nb_processed % 5000 == 0:
            logger.info("  Nb strings processed: {}".format(nb_processed))
    input_ids = torch.tensor(input_ids, dtype=torch.long, requires_grad=False, device=opt.device)
    nb_tokens = torch.tensor(nb_tokens, dtype=torch.long, requires_grad=False, device=opt.device)    
    return input_ids, nb_tokens


def get_missing_inputs(opt, token_ids, nb_tokens, lang_id):
    """ Given a tensor of padded token ids and a tensor indicating the number of actual (non padding tokens) per example, return dict containing additional inputs needed to feed the transformer.
    
    Args:
    opt
    tokenizer
    token_ids: tensor shape (n, max_length)
    nb_tokens: tensor shape (n, 1)
    lang_id: integer ID of the language of the examples

    """

    nb_examples, max_length = token_ids.size()
    inputs = {}
    
    # Segment IDs
    if opt.encoder_type == 'bert':
        inputs["token_type_ids"] = torch.tensor([[SEGMENT_ID] * max_length] * nb_examples, dtype=torch.long, requires_grad=False, device=opt.device)
    else:
        inputs["token_type_ids"] = None
        
    # Language IDs
    if opt.encoder_type == 'xlm':
        inputs["langs"] = torch.tensor([[lang_id] * max_length] * nb_examples, dtype=torch.long, requires_grad=False, device=opt.device)
    else:
        inputs["langs"] = None
        
    # Attention mask
    attention_mask = []
    for i in range(nb_examples):
        nb_tok = nb_tokens[i]
        padding_length = opt.max_seq_length - nb_tok
        mask = [1] * nb_tok + [0 if MASK_PADDING_WITH_ZERO else 1] * padding_length
        attention_mask.append(mask)
    inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long, requires_grad=False, device=opt.device)

    return inputs


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

            
def rotate_checkpoints(save_total_limit, output_dir, checkpoint_prefix, use_mtime=False, verbose=False):
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
        if verbose:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
