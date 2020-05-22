""" SPON scoring module. """

import math
from copy import deepcopy
import torch
import torch.nn.functional as F
from transformers import XLMModel, BertModel

MODEL_CLASSES = {
    'bert': BertModel,
    'xlm': XLMModel,
}

class SPONScorer(torch.nn.Module):
    """ SPON scoring module. """

    def __init__(self, opt, pretrained_encoder=None, encoder_config=None):
        """ Init from a pretrained encoder or an encoder config (in which case an encoder is initialized). If you are going to load a pretrained state_dict, then you will probably want to provide a config.
        Args:
        - opt
        - pretrained_encoder
        - encoder_config
        
        """
        
        super(SPONScorer, self).__init__()
        # Check args
        if pretrained_encoder is None and encoder_config is None:
            raise ValueError("Either pretrained_encoder or encoder_config must be provided.")
        self.normalize_encodings = opt.normalize_encodings
        self.epsilon = torch.tensor(opt.spon_epsilon, dtype=torch.float32)
        
        # Make 2 copies of the pretrained model
        if pretrained_encoder is None:
            model_class = MODEL_CLASSES[opt.encoder_type]
            self.encoder = model_class(encoder_config)
        else:
            self.encoder = deepcopy(pretrained_encoder)
            
        # Check if we freeze the encoder
        for p in self.encoder.parameters():
            if opt.freeze_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # Dropout layer
        self.do_dropout = opt.dropout_prob > 0
        if self.do_dropout:
            self.dropout = torch.nn.Dropout(p=opt.dropout_prob, inplace=False)
                
        # Query transformation layer 
        self.hidden_dim = self.encoder.config.emb_dim
        self.output_layer_type = opt.output_layer_type
        if self.output_layer_type == 'base':
            weight_data = torch.randn(self.hidden_dim) * math.sqrt(6./self.hidden_dim)
            bias_data = torch.randn(self.hidden_dim) * math.sqrt(6./self.hidden_dim)
            self.projector_weights = torch.nn.Parameter(weight_data)
            self.projector_bias = torch.nn.Parameter(bias_data)
        elif self.output_layer_type == 'projection':
            self.projector = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.projector.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
        elif self.output_layer_type == 'highway':
            self.projector = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.proj_gate = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.projector.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
            self.proj_gate.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
            self.proj_gate.bias.data.fill_(-2.0)

            
    def encode(self, inputs):
        """ Encode strings (either queries or candidates).
        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """
        
        outputs = self.encoder(**inputs)
        encs = outputs[0] # The last hidden states are the first element of the tuple
        encs = encs[:,0,:] # Keep only the hidden state of BOS
        return encs

        
    def compute_distance_to_satisfaction(self, query_enc, cand_encs):
        """
        Score candidates wrt query. Return 1-D Tensor of scores.
        Args:
        - query_enc: Tensor (1D) 
        - cand_encs: Tensor (2D) 

        """
        # Inspect input tensors
        if len(query_enc.size()) != 1:
            raise ValueError("query_enc must be 1-D")
        hidden_dim = query_enc.size()[0]
        if len(cand_encs.size()) != 2:
            raise ValueError("cand_encs must be 2-D")
        nb_cands, cand_hidden_dim = cand_encs.size()
        assert hidden_dim == cand_hidden_dim

        # Normalize encodings
        if self.normalize_encodings:
            query_enc = query_enc / torch.norm(query_enc, p=2) 
            cand_encs = cand_encs / torch.norm(cand_encs, p=2, dim=1, keepdim=True)
        
        # Apply dropout
        if self.do_dropout:
            query_enc = self.dropout(query_enc)
            cand_encs = self.dropout(cand_encs)

        # Project query
        query_enc = query_enc.unsqueeze(0)        
        if self.output_layer_type == 'base':
            query_enc = F.relu(query_enc * self.projector_weights + self.projector_bias) + query_enc
        elif self.output_layer_type == 'projection':
            query_enc = F.relu(self.projector(query_enc)) + query_enc
        elif self.output_layer_type == 'highway':
            proj = F.relu(self.projector(query_enc))
            gate = torch.sigmoid(self.proj_gate(query_enc))
            query_enc = (gate * proj) + ((1-gate) * query_enc)

        # Compute distance from satisfaction
        logits = torch.sum(torch.clamp(query_enc - cand_encs + self.epsilon, min=0), dim=1)
        return logits

    
    def logits_to_probs(self, logits, dim=None):
        """ Convert logits (aka distance to satisfaction scores) for a single query to probabilities. 
        Args:
        - logits: tensor of logits for a single query

        """
        nb_axes = len(logits.size())
        if nb_axes == 0 or nb_axes > 2:
            raise ValueError("Expected a 1-D tensor (or 2-D with a singleton dimension)")
        if nb_axes == 2 and dim is None:
            raise ValueError("dim along which we normalize must be specified if input tensor is 2-D")
        # Exponentiate and normalize, using a trick based on shift invariance (i.e. subtracting the max value) to avoid numerical overflow
        if nb_axes == 1:
            max_logit = torch.max(logits)
            exp = torch.exp(logits - max_logit)
            sum_exp = torch.sum(exp)
            return exp / sum_exp
        else:
            dimwise_max_vals, _ = torch.max(logits, dim=dim, keepdim=True)
            exp = torch.exp(logits - dimwise_max_vals)
            sum_exp = torch.sum(exp, dim=dim, keepdim=True)
            return exp / sum_exp

    
    def forward(self, query_inputs, cand_inputs, convert_logits_to_probs=False):
        """ Forward pass from encodings to scores.
        Args:
        - query_inputs: dict containing query_enc (1-D)
        - cand_inputs: dict containing cand_encs (2-D)

        """
        logits = self.compute_distance_to_satisfaction(query_inputs["query_enc"], cand_inputs["cand_encs"])
        if convert_logits_to_probs:
            return self.logits_to_probs(logits)
        else:
            return logits


