""" Scoring module. """

import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import XLMModel, BertModel


MODEL_CLASSES = {
    'bert': BertModel,
    'xlm': XLMModel,
}


def softmax(logits):
    """ Exponentiate and normalize un-normalized scores (i.e. compute softmax) for one or more queries. Numerically stable.x
    
    Args:
    - logits: 1-D tensor of logits for a single query or 2-D tensor where each *row* contains the logits of a query.

    """
    nb_axes = len(logits.size())
    assert nb_axes in [1, 2]
    if nb_axes == 1:
        softmax = torch.exp(logits - torch.logsumexp(logits, 0))
    elif nb_axes == 2:
        softmax = torch.exp(logits - torch.logsumexp(logits, 1, keepdim=True))
    return softmax


class Scorer(torch.nn.Module):
    """ Scoring module. """

    def __init__(self, opt, pretrained_encoder=None, encoder_config=None):
        """ Init from a pretrained encoder or an encoder config (in which case
        an encoder is initialized). If you are going to load a
        pretrained state_dict, then you will probably want to provide
        a config.

        Args:
        - pretrained_encoder
        - encoder_config

        """
        super(Scorer, self).__init__()

        # Check args
        if pretrained_encoder is None and encoder_config is None:
            raise ValueError("Either pretrained_encoder or encoder_config must be provided.")
        self.encoding_arch = opt.encoding_arch
        if self.encoding_arch not in ['single_q', 'single_c', 'bi_q', 'bi_c']:
            raise ValueError("Unrecognized encoding architecture '%s'" % opt.encoding_arch)
        self.freeze_encoder = opt.freeze_encoder
        self.normalize_encodings = opt.normalize_encodings
        self.dropout_prob = opt.dropout_prob
        self.transform = opt.transform
        # Do we transform query encodings or candidate encodings?
        self.transform_queries = self.encoding_arch in ['single_q', 'bi_q']
        if self.transform not in ['none', 'scaling', 'projection', 'highway']:
            raise ValueError("Unrecognized transform '%s'" % opt.transform)
        self.score_fn = opt.score_fn
        if self.score_fn not in ['dot', 'spon']:
            raise ValueError("Unrecognized score function '%s'" % opt.score_fn)
        if self.score_fn == "spon":
            self.spon_epsilon = torch.tensor(opt.spon_epsilon, dtype=torch.float32)
            
        # Make encoder
        if self.encoding_arch in ["single_q", "single_c"]:
            self.encoder = SingleEncoder(opt, pretrained_encoder=pretrained_encoder, encoder_config=encoder_config)
        else:
            self.encoder = BiEncoder(opt, pretrained_encoder=pretrained_encoder, encoder_config=encoder_config)
        self.hidden_dim = self.encoder.encoder_config.emb_dim

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=opt.dropout_prob, inplace=False)
                
        # Transformation layer
        if self.transform == 'scaling':
            weight_data = torch.randn(self.hidden_dim) * math.sqrt(6./self.hidden_dim)
            bias_data = torch.randn(self.hidden_dim) * math.sqrt(6./self.hidden_dim)
            self.projector_weights = torch.nn.Parameter(weight_data)
            self.projector_bias = torch.nn.Parameter(bias_data)
        elif self.transform == 'projection':
            self.projector = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.projector.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
        elif self.transform == 'highway':
            self.projector = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.proj_gate = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.projector.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
            self.proj_gate.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
            self.proj_gate.bias.data.fill_(-2.0)

            
    def encode_queries(self, inputs):
        """ Encode queries.

        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """
        encs = self.encoder.encode_queries(inputs)
        return encs

    
    def encode_candidates(self, inputs):
        """ Encode candidates.

        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """
        encs = self.encoder.encode_candidates(inputs)
        return encs

    
    def compute_distance_to_satisfaction(self, query_enc, cand_encs):
        """ Score candidates wrt query. Return 1-D Tensor of scores.

        Args:
        - query_enc: Tensor (1D) 
        - cand_encs: Tensor (2D) 

        """        
        dist = torch.sum(torch.clamp(query_enc - cand_encs + self.spon_epsilon, min=0), dim=1)
        return dist


    def compute_dot_product(self, query_enc, cand_encs):
        """  Score candidates wrt query by taking the dot product. Return 1-D Tensor of scores.

        Args:
        - query_enc: Tensor (1D) 
        - cand_encs: Tensor (2D) 

        """        
        dot = torch.matmul(query_enc, cand_encs.permute(1,0)).squeeze(0)
        return dot


    def transform_encodings(self, encodings):
        """ Apply transformation layer to query encodings or candidate encodings.

        Args:
        - encodings: tensor of encodings. 1-D for queries, 2-D for candidates. If it is 2-D, each *row* must contain the encoding of a candidate.

        """
        nb_axes = len(encodings.size())
        assert nb_axes in [1,2]
        if nb_axes == 1:
            encodings = encodings.unsqueeze(0)                    
        if self.transform == 'scaling':
            encodings = F.relu(encodings * self.projector_weights + self.projector_bias) + encodings
        elif self.transform == 'projection':
            encodings = F.relu(self.projector(encodings)) + encodings
        elif self.transform == 'highway':
            proj = F.relu(self.projector(encodings))
            gate = torch.sigmoid(self.proj_gate(encodings))
            encodings = (gate * proj) + ((1-gate) * encodings)
        return encodings

            
    def score_candidates(self, query_enc, cand_encs):
        """
        Score candidates wrt query. Return 1-D Tensor of scores.

        Args:
        - query_enc: Tensor (1D)
        - cand_encs: Tensor (2D)

	Returns: 1-D tensor of un-normalized candidate scores for this query.

        """
        # Inspect input tensors
        if len(query_enc.size()) != 1:
            raise ValueError("query_enc must be 1-D")
        assert query_enc.size()[0] == self.hidden_dim
        if len(cand_encs.size()) != 2:
            raise ValueError("cand_encs must be 2-D")
        nb_cands, cand_hidden_dim = cand_encs.size()
        assert cand_hidden_dim == self.hidden_dim

        # Apply dropout
        if self.dropout_prob > 0:
            query_enc = self.dropout(query_enc)
            cand_encs = self.dropout(cand_encs)

        # Normalize encodings
        if self.normalize_encodings:
            query_enc = query_enc / torch.norm(query_enc, p=2) 
            cand_encs = cand_encs / torch.norm(cand_encs, p=2, dim=1, keepdim=True)
            
        # Apply transformation to query or candidate encodings
        if self.transform_queries:
            query_enc = self.transform_encodings(query_enc)
        else:
            cand_encs = self.transform_encodings(cand_encs)
            
        # Check for NaN
        #if torch.isnan(query_enc).any():
        #    query_enc = query_enc.masked_fill(torch.isnan(query_enc), 0)
        #if torch.isnan(cand_encs).any():
        #    cand_encs = cand_encs.masked_fill(torch.isnan(cand_encs), 0)
        assert not torch.isnan(query_enc).any()
        assert not torch.isnan(cand_encs).any()        

        # Compute scores
        if self.score_fn == 'dot':
            scores = self.compute_dot_product(query_enc, cand_encs)
        elif self.score_fn == 'spon':
            scores = self.compute_distance_to_satisfaction(query_enc, cand_encs)

        # Check for NaN
        assert not torch.isnan(scores).any()
        return scores


    def convert_logits_to_probs(self, logits):
        return softmax(logits)

    
    def forward(self, query_inputs, cand_inputs, convert_logits_to_probs=False):
        """ Forward pass from encodings to scores.

        Args:
        - query_inputs: dict containing query_enc (1-D)
        - cand_inputs: dict containing cand_encs (2-D)

        """
        logits = self.score_candidates(query_inputs["query_enc"], cand_inputs["cand_encs"])        
        if convert_logits_to_probs:
            return self.convert_logits_to_probs(logits)
        else:
            return logits

    
    
class BiEncoder(torch.nn.Module):
    """ Bi-encoder module. """

    def __init__(self, opt, pretrained_encoder=None, encoder_config=None):
        """ Init from a pretrained encoder or an encoder config (in which case
        an encoder is initialized). If you are going to load a
        pretrained state_dict, then you will probably want to provide
        a config.

        Args:
        - pretrained_encoder
        - encoder_config

        """
        super(BiEncoder, self).__init__()

        # Check args
        if pretrained_encoder is None and encoder_config is None:
            raise ValueError("Either pretrained_encoder or encoder_config must be provided.")
        
        # Make 2 copies of the pretrained model
        if pretrained_encoder is None:
            model_class = MODEL_CLASSES[opt.encoder_type]
            self.encoder_q = model_class(encoder_config)
            self.encoder_c = deepcopy(self.encoder_q)
        else:
            self.encoder_q = deepcopy(pretrained_encoder)
            self.encoder_c = deepcopy(pretrained_encoder)
            
        # Check if we freeze the encoders
        self.freeze_cand_encoder = opt.freeze_encoder
        self.freeze_query_encoder = opt.freeze_encoder
        for p in self.encoder_c.parameters():
            if self.freeze_cand_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True
        for p in self.encoder_q.parameters():
            if self.freeze_query_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # Store encoder confi
        self.encoder_config = self.encoder_q.config
        
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
        
        
    def forward(self, inputs):
        """ Dummy forward function. """
        return


    
class SingleEncoder(torch.nn.Module):
    """ Single encoder module. """

    def __init__(self, opt, pretrained_encoder=None, encoder_config=None):
        """ Init from a pretrained encoder or an encoder config (in which case
        an encoder is initialized). If you are going to load a
        pretrained state_dict, then you will probably want to provide
        a config.

        Args:
        - opt
        - pretrained_encoder
        - encoder_config

        """
        super(SingleEncoder, self).__init__()

        # Check args
        if pretrained_encoder is None and encoder_config is None:
            raise ValueError("Either pretrained_encoder or encoder_config must be provided.")
        
        # Encoder
        if pretrained_encoder is None:
            model_class = MODEL_CLASSES[opt.encoder_type]
            self.encoder = model_class(encoder_config)
        else:
            self.encoder = deepcopy(pretrained_encoder)
            
        # Check if we freeze the encoder
        self.freeze_encoder = opt.freeze_encoder
        for p in self.encoder.parameters():
            if self.freeze_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # Store encoder confi
        self.encoder_config = self.encoder.config
                
    def encode(self, inputs):
        """ Encode strings (either queries or candidates).

        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """        
        outputs = self.encoder(**inputs)
        encs = outputs[0] # The last hidden states are the first element of the tuple
        encs = encs[:,0,:] # Keep only the hidden state of BOS
        return encs

    
    def encode_queries(self, inputs):
        return self.encode(inputs)

    
    def encode_candidates(self, inputs):
        return self.encode(inputs)

    
    def forward(self, inputs):
        """ Dummy forward function """
        return 

