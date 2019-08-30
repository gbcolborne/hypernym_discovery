from __future__ import division
import codecs
from copy import deepcopy
import numpy as np
import torch
from utils import denormalize_term, wrap_in_var
from task9_scorer import average_precision

class Evaluator:
    NB_PRED = 15
    BATCH_NB_CANDIDATES = 5000

    def __init__(self, model, query_embed, query_cand_ids):
        """
        Args:
        - model: model
        - query_embed: Embedding object containing query embeddings
        - query_cand_ids: list containing the candidate ID of each
          query (None if query is not also a candidate), in same order
          as the query embeddings.

        """

        self.model = model
        self.query_embed = query_embed
        self.nb_queries = self.query_embed.weight.shape[0]
        self.query_cand_ids = query_cand_ids
        if len(query_cand_ids) != self.nb_queries:
            msg = "Error: number of query embeddings ({}) ".format(self.nb_queries)
            msg += "does not match number of query candidate IDs ({})".format(len(query_cand_ids))
            raise ValueError(msg)

        # Create list of torch Variables containing a batch of
        # candidates, shape (1,BATCH_NB_CANDIDATES).
        self.nb_candidates = self.model.get_nb_candidates()
        self.candidate_batches = []
        nb_batches = self.nb_candidates // self.BATCH_NB_CANDIDATES
        if self.nb_candidates % self.BATCH_NB_CANDIDATES:
            nb_batches += 1
        for batch_ix in range(nb_batches):
            start = batch_ix * self.BATCH_NB_CANDIDATES
            end = (batch_ix + 1) * self.BATCH_NB_CANDIDATES
            if end > self.nb_candidates:
                end = self.nb_candidates
            batch_data = torch.tensor(list(range(start, end)), dtype=torch.int64).unsqueeze(0)
            batch_var = wrap_in_var(batch_data, False, cuda=model.use_cuda)
            self.candidate_batches.append(deepcopy(batch_var))

        # Create list of torch Variables containing a query ID
        self.query_ids = []
        for i in range(self.nb_queries):
            self.query_ids.append(wrap_in_var(torch.tensor([i], dtype=torch.int64), False, cuda=model.use_cuda))

    def set_model(self, model):
        self.model = model

    def _get_candidate_scores(self, query_ix):
        """ 
        Given a query, get scores of all candidates.

        Args:
        - query_ix: index of query
        
        Return:
        - 1-D numpy array of scores, where the ith element is the
          score of the ith candidate in self.candidate_list

        """
        
        score_batches = []
        for candidate_batch in self.candidate_batches:
            scores = self.model(self.query_embed(self.query_ids[query_ix]), candidate_batch)
            score_batches.append(scores)
        scores = torch.cat(score_batches, 1).squeeze(0)
        scores = scores.data
        if self.model.use_cuda:
            scores = scores.cpu()
        scores = scores.numpy()

        # If query is also a candidate, set it's score to -inf
        cand_id = self.query_cand_ids[query_ix]
        if cand_id:
            scores[cand_id] = float("-inf")           
        return scores
    

    def _get_top_candidates(self, query_ix, n):
        """
        Given a query, get top n candidates.

        Args:
        - query_ix: index of query
        - n: number of candidates to return

        Return:
        - list of top n candidates, sorted by score in reverse order
        
        """
        scores = self._get_candidate_scores(query_ix)
        top_candidates = scores.argsort()[-1:-(n+1):-1]
        return top_candidates

    def get_MAP(self, gold_ids):
        """ Compute mean average precision of predicted hypernyms of
        queries with respect to gold hypernyms.

        Args:
        - gold_ids: list of sets of gold hypernym candidate IDs, one
          list for each query, in same order as the query
          embeddings. May be None (if Evaluator is noy used to compute
          evaluation metrics, only to predict hypernyms)

        Returns:
        - MAP

        """
        self.model.eval()
        AP_vals = []
        for q_ix in range(self.nb_queries):
            h_ids = gold_ids[q_ix]
            n = min(len(h_ids), self.NB_PRED)
            pred_ids = self._get_top_candidates(q_ix, n)
            # Create array indicating correctness of the top 15
            # predictions, in the same way as the scorer of
            # SemEval-2018 Task 9.
            is_correct = [0 for _ in range(self.NB_PRED)]
            for i, pred_id in enumerate(pred_ids):
                if pred_id in h_ids:
                    is_correct[i] = 1
            AP_vals.append(average_precision(is_correct, len(h_ids)))
        MAP = np.mean(AP_vals)
        return MAP

    def write_predictions(self, path, candidates):
        """ Write predictions of model on queries.

        Args:
        - path: path of output file
        - candidates: list of candidates (in same order as their embeddings in the model)

        """
        self.model.eval()
        with codecs.open(path, "w", encoding="utf-8") as f:
            for q_ix in range(self.nb_queries):
                pred_ids = self._get_top_candidates(q_ix, self.NB_PRED)
                pred_strings = [denormalize_term(candidates[p]) for p in pred_ids]
                f.write("\t".join(pred_strings) + "\n")
