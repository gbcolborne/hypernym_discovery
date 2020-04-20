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

    def encode_candidates(self, inputs):
        """ Encode candidates.
        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """

        outputs = self.encoder_c(**inputs)
        encs = outputs[0] # The last hidden states are the first element of the tuple
        encs = encs[:,0,:] # Keep only the hidden state of BOS
        return encs

    def encode_queries(self, inputs):
        """ Encode queries.
        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """

        outputs = self.encoder_q(**inputs)
        encs = outputs[0] # The last hidden states are the first element of the tuple
        encs = encs[:,0,:] # Keep only the hidden state of BOS        
        return encs
        
        
    def score_candidates(self, query_encs, cand_encs):
        """
        Score pairs of query and candidate encodings by taking the cosine. Return 1-D Tensor of scores.
        Args:
        - query_encs: Tensor (1D ro 2D)
        - cand_encs: Tensor (1D or 2D) *Note: if both tensors are 2D, their shapes must match, as we perform a batch dot product on pairs of encodings.

        """
        # Inspect input tensors
        nb_axes = len(query_encs.size())
        if nb_axes == 1:
            nb_queries = 1
            hidden_dim = query_encs.size()[0]
        elif nb_axes == 2:
            nb_queries, hidden_dim = query_encs.size()
        else:
            raise ValueError("query_encs must be 1-D or 2-D")
        nb_axes = len(cand_encs.size())
        if nb_axes == 1:
            cand_hidden_dim = cand_encs.size()[0]
        elif nb_axes == 2:
            nb_cands, cand_hidden_dim = cand_encs.size()
        else:
            raise ValueError("cand_encs must be 1-D or 2-D")
        assert hidden_dim == cand_hidden_dim
        if nb_queries > 1 and nb_cands > 1:
            if nb_queries != nb_cands:
                msg = "nb_queries and nb_cands must match if both tensors are 2D (in this case, we perform a batch dot product on pairs of encodings)"
                raise ValueError(msg)

        # Normalize
        if nb_queries > 1:
            query_encs_norm = query_encs / torch.norm(query_encs, p=2, dim=1, keepdim=True)            
        else:
            query_encs_norm = query_encs / torch.norm(query_encs, p=2)
        if nb_cands > 1:
            cand_encs_norm = cand_encs / torch.norm(cand_encs, p=2, dim=1, keepdim=True)
        else:
            cand_encs_norm = cand_encs / torch.norm(cand_encs, p=2)
            
        # Compute dot product
        if nb_queries > 1:
            if nb_cands > 1:
                scores = torch.bmm(query_encs_norm.unsqueeze(1), cand_encs_norm.unsqueeze(2)).squeeze(2).squeeze(1)
            else:
                scores = torch.matmul(cand_encs_norm, query_encs_norm.permute(1,0)).unsqueeze(0)
        else:
            if nb_cands > 1:
                scores = torch.matmul(query_encs_norm, cand_encs_norm.permute(1,0))
            else:
                scores = torch.matmul(query_encs_norm, cand_encs_norm.permute(1,0)).unsqueeze(0)
        return scores

    def forward(self, query_inputs, cand_inputs):
        """ Forward pass from encodings to scores.
        Args:
        - query_inputs: dict containing query_encs (1-D or 2-D Tensor)
        - cand_inputs: dict containing cand_encs (1-D or 2-D Tensor)

        """
        return self.score_candidates(query_inputs["query_encs"], cand_inputs["cand_encs"])

