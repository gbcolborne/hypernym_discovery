""" Bi-encoder scoring module. """

from copy import deepcopy
import torch

class BiEncoderScorer(torch.nn.Module):
    """ Bi-encoder scoring module. """

    def __init__(self, opt, pretrained_encoder):
        super(BiEncoderScorer, self).__init__()

        # Make 2 copies of the pretrained model
        self.encoder_q = deepcopy(pretrained_encoder)
        self.encoder_c = deepcopy(pretrained_encoder)

        # Check if we freeze the candidate encoder
        if opt.freeze_cand_encoder:
            self.encoder_c.require_grad = False
        else:
            self.encoder_c.require_grad = True
        self.encoder_q.requires_grad = True

        # Define loss
        self.loss = torch.nn.CrossEntropyLoss()

    def encode_candidates(self, inputs):
        """ Encode candidates.
        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """
        # Transformers take can not handle 3D inputs, so we will iterate over one axis of the tensor
        input_ids = inputs["input_ids"]
        (nb_queries, per_query_nb_examples, max_length) = input_ids.size()
        print("Nb queries: {}".format(nb_queries))
        print("Per query nb examples: {}".format(per_query_nb_examples))
        print("Max length: {}".format(max_length))
        
        cand_encs = []
        for i in range(nb_queries):
            print(i)
            inputs_sub = {}
            for key in inputs:
                if inputs[key] == None:
                    inputs_sub[key] = None
                else:
                    inputs_sub[key] = inputs[key][i]
            outputs = self.encoder_c(**inputs_sub)
            encs = outputs[0] # The last hidden states are the first element of the tuple
            cand_encs.append(encs)
        return torch.cat(cand_encs)

    def encode_queries(self, inputs):
        """ Encode queries.
        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """

        outputs = self.encoder_q(**inputs)
        query_encs = outputs[0] # The last hidden states are the first element of the tuple
        return query_encs
        
        
    def score_candidates(self, query_inputs, cand_inputs):
        """
        Score (query, candidate) pairs by encoding queries and candidates and taking the dot product.
        Args:
        - query_inputs: dict containing query inputs. The query inputs can be either query_encs (dims = [nb queries, hidden dim]) if the query encodings were pre-computed, or the following if the queries should be encoded on the fly: input_ids, attention_mask, token_type_ids, and langs (dims = [nb queries, max length]). 
        - cand_inputs: dict containing candate inputs, as for queries.

        """

        # Encode queries
        if 'query_encs' in query_inputs:
            query_encs = query_inputs['query_encs']
        else:
            query_encs = self.encode_queries(query_inputs)

        # Encode candidates 
        if 'cand_encs' in cand_inputs:
            cand_encs = cand_inputs['cand_encs']
        else:
            cand_encs = self.encode_candidates(cand_inputs)
        
        # Compute dot product
        scores = torch.bmm(query_encs.unsqueeze(1), cand_encs.transpose(1, 2)).squeeze(1)
        return scores

    def compute_loss(self, logits, targets):
        """ Compute loss.
        Args:
        - logits: unnormalized class scores. Shape is (N,C) where C = number of classes, or (N, C, d_1, d_2, ..., d_K) with K >= 1 in the case of K-dimensional loss.
        - targets: targets. (N) where each value is 0 <= targets[i] <= C-1, or (N, d_1, d_2, ..., d_K) with K >= 1 in the case of K-dimensional loss.

        """

        return self.loss(logits, targets)

    def forward(self, query_inputs, cand_inputs):
        return self.score_candidates(query_inputs, cand_inputs)

