""" Bi-encoder scoring module. """

import math
from copy import deepcopy
import torch
from transformers import XLMModel, BertModel

MODEL_CLASSES = {
    'bert': BertModel,
    'xlm': XLMModel,
}

ADD_EYE_TO_INIT = True
RELU_AFTER_PROJECTION = False

class BiEncoderScorer(torch.nn.Module):
    """ Bi-encoder scoring module. """

    def __init__(self, opt, pretrained_encoder=None, encoder_config=None):
        """ Init from a pretrained encoder or an encoder config (in which case an encoder is initialized). If you are going to load a pretrained state_dict, then you will probably want to provide a config.
        Args:
        - pretrained_encoder
        - encoder_config
        
        """
        
        super(BiEncoderScorer, self).__init__()
        # Check args
        if pretrained_encoder is None and encoder_config is None:
            raise ValueError("Either pretrained_encoder or encoder_config must be provided.")
        self.normalize_encodings = opt.normalize_encodings
        self.add_eye_to_init = ADD_EYE_TO_INIT
        self.relu_after_projection = RELU_AFTER_PROJECTION
        
        # Make 2 copies of the pretrained model
        if pretrained_encoder is None:
            model_class = MODEL_CLASSES[opt.encoder_type]
            self.encoder_q = model_class(encoder_config)
            self.encoder_c = deepcopy(self.encoder_q)
        else:
            self.encoder_q = deepcopy(pretrained_encoder)
            self.encoder_c = deepcopy(pretrained_encoder)
            
        # Check if we freeze the candidate encoder
        for p in self.encoder_c.parameters():
            if opt.freeze_cand_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True
        for p in self.encoder_q.parameters():
            if opt.freeze_query_encoder:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # Dropout layer
        self.do_dropout = opt.dropout_prob > 0
        if self.do_dropout:
            self.dropout = torch.nn.Dropout(p=opt.dropout_prob, inplace=False)
                
        # Projection layer 
        self.hidden_dim = self.encoder_q.config.emb_dim
        self.projector_q = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.projector_c = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        # Xavier init
        self.projector_q.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
        self.projector_c.weight.data = torch.randn(self.hidden_dim, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + self.hidden_dim))
        if self.add_eye_to_init:
            self.projector_q.weight.data = self.projector_q.weight.data + torch.eye(self.hidden_dim, self.hidden_dim)
            self.projector_c.weight.data = self.projector_c.weight.data + torch.eye(self.hidden_dim, self.hidden_dim)

        # Candidate bias layer
        self.cand_bias = torch.nn.Linear(in_features=self.hidden_dim, out_features=1, bias=False)
        # Xavier init
        self.cand_bias.weight.data = torch.randn(1, self.hidden_dim)*math.sqrt(6./(self.hidden_dim + 1))
        
        # Output layer
        self.output = torch.nn.Linear(in_features=2, out_features=1, bias=True)
        # Init with ones
        torch.nn.init.ones_(self.output.weight.data)
            
        
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
        
        
    def score_candidates(self, query_enc, cand_encs):
        """
        Score candidates wrt query by taking the dot product. Return 1-D Tensor of scores.
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

        # Project
        query_enc = query_enc.unsqueeze(0)
        query_enc = self.projector_q(query_enc)
        cand_encs = self.projector_c(cand_encs)
        if self.relu_after_projection:
            query_enc = query_enc.clamp(min=0)
            cand_encs = cand_encs.clamp(min=0)
            
        # Apply dropout 
        if self.do_dropout:
            query_enc = self.dropout(query_enc)
            cand_encs = self.dropout(cand_encs)
            
        # Normalize
        if self.normalize_encodings:
            query_enc = query_enc / torch.norm(query_enc, p=2)
            cand_encs = cand_encs / torch.norm(cand_encs, p=2, dim=1, keepdim=True)
            
        # Compute dot product
        dot = torch.matmul(query_enc, cand_encs.permute(1,0)).squeeze(1)
        
        # Compute candidate bias
        cand_bias = self.cand_bias(cand_encs)

        # Compute output logits
        inputs = torch.cat([dot.T,cand_bias], dim=1)
        logits = self.output(inputs).squeeze(1)
        return logits

    def forward(self, query_inputs, cand_inputs):
        """ Forward pass from encodings to scores.
        Args:
        - query_inputs: dict containing query_enc (1-D)
        - cand_inputs: dict containing cand_encs (2-D)

        """
        return self.score_candidates(query_inputs["query_enc"], cand_inputs["cand_encs"])

