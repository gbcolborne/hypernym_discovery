""" SPON scoring module. """

import math
from copy import deepcopy
import torch
from transformers import XLMModel, BertModel

MODEL_CLASSES = {
    'bert': BertModel,
    'xlm': XLMModel,
}

ADD_EYE_TO_INIT = True

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
        self.add_eye_to_init = ADD_EYE_TO_INIT

        # Check args
        if pretrained_encoder is None and encoder_config is None:
            raise ValueError("Either pretrained_encoder or encoder_config must be provided.")
        self.normalization_factor = opt.normalization_factor
        self.epsilon = torch.tensor(opt.spon_epsilon, dtype=torch.float32)
        self.zero = torch.tensor(0, dtype=torch.float32)
        
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
                
        # Query projection layer 
        self.hidden_dim = self.encoder.config.emb_dim
        self.use_projection_matrix = opt.use_projection_matrix
        if self.use_projection_matrix:
            self.projector = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            # Xavier init
            self.projector.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
            if self.add_eye_to_init:
                self.projector.weight.data = self.projector.weight.data + torch.eye(self.hidden_dim, self.hidden_dim)
        else:
            weight_data = torch.randn(self.hidden_dim) * math.sqrt(6./self.hidden_dim)
            bias_data = torch.randn(self.hidden_dim) * math.sqrt(6./self.hidden_dim)
            self.projector_weights = torch.nn.Parameter(weight_data)
            self.projector_bias = torch.nn.Parameter(bias_data)
            
        
    def encode(self, inputs):
        """ Encode candidates.
        Args:
        - inputs: dict containing input_ids, attention_mask, token_type_ids, langs

        """
        
        outputs = self.encoder(**inputs)
        encs = outputs[0] # The last hidden states are the first element of the tuple
        encs = encs[:,0,:] # Keep only the hidden state of BOS
        return encs

        
    def score_candidates(self, query_enc, cand_encs):
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

        # Apply dropout
        query_enc = query_enc.unsqueeze(0)        
        if self.do_dropout:
            query_enc = self.dropout(query_enc)
            cand_encs = self.dropout(cand_encs)

        # Project query
        if self.use_projection_matrix:
            query_enc = self.projector(self.query_enc)
        else:
            query_enc = query_enc * self.projector_weights + self.projector_bias

        # Soft length normalization
        if self.normalization_factor > 0.0:
            query_enc = query_enc / ((1-self.normalization_factor) * torch.norm(query_enc, p=2) + self.normalization_factor)
            cand_encs = cand_encs / ((1-self.normalization_factor) * torch.norm(cand_encs, p=2, dim=1, keepdim=True) + self.normalization_factor)

        # Compute distance from satisfaction
        logits = torch.sum(torch.max(query_enc - cand_encs + self.epsilon, self.zero))
        return logits

    def forward(self, query_inputs, cand_inputs):
        """ Forward pass from encodings to scores.
        Args:
        - query_inputs: dict containing query_enc (1-D)
        - cand_inputs: dict containing cand_encs (2-D)

        """
        return self.score_candidates(query_inputs["query_enc"], cand_inputs["cand_encs"])

