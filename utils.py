import codecs
from copy import deepcopy
import numpy as np
import torch
from torch import autograd
from torch import nn


def make_embedder(embeds, grad=False, cuda=False, sparse=False):
    """ Make an Embedding module from a numpy array.
    Args:
    - embeds: numpy array of embeddings, shape (nb embeddings, embedding size)

    Keyword args:
    - requires_grad: boolean which specifies whether grad is required 
    - cuda: boolean which specifies whether we use cuda 
    - sparse: boolean which specifies whether embeddings are sparse 
    
    """
    nb_embeds, dim = embeds.shape
    embed = nn.Embedding(nb_embeds, dim, sparse=sparse)
    weights = deepcopy(embeds)
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    weight_data = torch.tensor(weights, dtype=torch.float32, device=device)
    embed.weight=nn.Parameter(weight_data)
    embed.weight.requires_grad = grad
    return embed

def make_embedding_matrix(word2vec, words, seed=0):
    """ Given a mapping of words to vectors and a list of words, make
    a matrix containing the vector of each word. Assign a random
    vector to words that don't have one.

    Args:
    - word2vec: dict that maps words to vectors
    - words: list of words
    - seed: seed for numpy's RNG

    Returns:
    - matrix: containing row vector for each word in same order as the
      list of words

    """
    np.random.seed(seed)
    dim = word2vec[words[0]].shape[0]
    dtype = word2vec[words[0]].dtype
    matrix = np.zeros((len(words), dim), dtype=dtype)
    for (i,word) in enumerate(words):
        if word in word2vec:
            matrix[i] = word2vec[word]
        else:
            matrix[i] = np.random.uniform(low=-0.5, high=0.5) / dim
    return matrix


def normalize_numpy_matrix(x):
    """ Make rows in a 2-D numpy array unit-length. """
    return x / np.linalg.norm(x, ord=2, axis=1).reshape(-1,1)

def get_embeddings(path, dtype=np.float32):
    """Get word embeddings from text file. 

    Args:
    - path
    - dtype: dtype of matrix

    Returns:
    - vocab (list of words) 
    - dict that maps words to vectors

    """
    # Get vector size
    with codecs.open(path, "r", "utf-8") as f:
        elems = f.readline().strip().split()
        if len(elems) == 2:
            header = True
            dim = int(elems[1])
        else:
            header = False
            dim = len(elems)-1
    words = []
    word2vec = {}
    with codecs.open(path, "r", "utf-8") as f:
        line_count = 0
        if header:
            f.readline()
            line_count = 1
        for line in f:
            line_count += 1
            elems = line.strip().split()
            if len(elems) == dim + 1:
                word = elems[0]
                try:
                    vec = np.asarray(elems[1:], dtype=dtype)
                    words.append(word)
                    word2vec[word] = vec
                except ValueError as e:
                    print("ValueError: Skipping line {}".format(line_count))
            else:
                msg = "Error: Skipping line {}. ".format(line_count)
                msg += "Expected {} elements, found {}.".format(dim+1, len(elems))
                print()
    return words, word2vec

def expand_subtask_name(subtask):
    """ Given short name of subtask, return long name. """
    if subtask == "1A":
        return "1A.english"
    elif subtask == "1B":
        return "1B.italian"
    elif subtask == "1C":
        return "1C.spanish"
    elif subtask == "2A":
        return "2A.medical"
    elif subtask == "2B":
        return "2B.music"
    else:
        msg = "Unrecognized subtask name '{}'".format(subtask)
        raise ValueError(msg)

def load_vocab(path_data, subtask, lower_queries=False):
    """ Given the path of the directory containing the data for
    SemEval 2018 task 9 and the name of a subtask (e.g. 1A), load
    candidates and queries (training, trial, and test). Return set of
    candidates and set of queries.
    """
    dataname = expand_subtask_name(subtask)
    path = "{}/vocabulary/{}.vocabulary.txt".format(path_data, dataname)
    candidates = set(load_candidates(path, normalize=False))
    path = "{}/vocabulary/{}.vocabulary.txt".format(path_data, dataname)
    queries = set()
    for part in ["training", "trial", "test"]:
        path = "{}/{}/data/{}.{}.data.txt".format(path_data, part, dataname, part)
        q, _ = load_queries(path, normalize=False)
        if lower_queries:
            q = [query.lower() for query in q]
        queries.update(q)
    return candidates, queries

def load_queries(path, normalize=True):
    """Given the path of a query file, return list of queries and list
    of query types.

    """
    with codecs.open(path, "r", encoding="utf-8") as f:
        queries = []
        query_types = []
        for line in f:
            q, qtype = line.strip().split("\t")
            if normalize:
                q = normalize_term(q)
            queries.append(q)
            query_types.append(qtype)
    return queries, query_types

def load_hypernyms(path, normalize=True):
    """Given the path of a hypernyms file, return list of lists of
    hypernyms.

    """
    with codecs.open(path, "r", encoding="utf-8") as f:
        hypernyms = []
        for line in f:
            h_list = line.strip().split("\t")
            if normalize:
                h_list = [normalize_term(h) for h in h_list]
            hypernyms.append(h_list)
    return hypernyms

def load_candidates(path, normalize=True):
    """Given the path of a list of candidate hypernyms, return list of
    candidates.

    """
    with codecs.open(path, "r", encoding="utf-8") as f:
        candidates = []
        for line in f:
            c = line.strip()
            if len(c):
                if normalize:
                    c = normalize_term(c)
                candidates.append(c)
        return candidates

def denormalize_term(candidate):
    """ Reverse normalization of candidate by replacing underscores
    with spaces."""
    return candidate.replace("_", " ")

def normalize_term(term):
    """ Normalize a query or candidate: lower-case (which only affects
    queries, as candidates are all lower-case) and replace spaces with
    underscores."""
    return term.lower().replace(" ", "_")

def print_params(model):
    """ Print parameters of a PyTorch model. """
    for name, param in model.named_parameters():
        device = "CPU"
        if param.device.type=="cuda":
            device = "GPU"
        grad = "no"
        if param.requires_grad:
            grad = "yes"
        msg = "- {} (on {}, grad={}) ".format(name, device, grad)    
        print(msg)

def write_queries_and_hypernyms(queries, hypernyms, path_queries, path_hypernyms, indices=None):
    """Given list of queries and corresponding list of lists of hypernyms,
    write queries and hypernyms in SemEval format. If a list of
    indices are provided, only the queries at those indices will be
    written (along with their hypernyms).

    """
    
    if indices is None:
        indices = range(len(queries))
    with open(path_queries, 'w') as fq, open(path_hypernyms, 'w') as fg:
        for i in indices:
            fq.write("{}\n".format(queries[i]))
            fg.write("{}\n".format("\t".join(hypernyms[i])))
    return None
