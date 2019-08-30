import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F

"""
Projection learning model for hypernym discovery.  See http://www.aclweb.org/anthology/S18-1116.
"""

class Classifier(nn.Module):
    """Classifier to be used with a Projector (for feature extraction)."""

    def __init__(self, *args, **kwargs):
        """
        Initialize model. 

        Args:
        - projector: an instance of Projector.

        Keyword args:
        - cuda: boolean which specifies whether we use cuda
        - seed: seed for RNG.

        """
        super(Classifier, self).__init__()

        # Process keyword args
        if "seed" in kwargs and kwargs["seed"]:
            torch.manual_seed(kwargs["seed"])
        if "cuda" not in kwargs:
            kwargs["cuda"] = False
        self.use_cuda = kwargs["cuda"]

        # Process positional args.
        if len(args) != 1:
            msg = "Expected 1 positional arg, found {}.".format(len(args))
            raise TypeError(msg)

        self.projector = args[0]
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.output = nn.Linear(self.projector.nb_proj, device=device)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=None, reduction="sum")


    def get_nb_projections(self):
        return self.projector.get_nb_projections()

    def get_nb_candidates(self):
        return self.projector.cand_embed.weight.shape[0]

    def get_dim(self):
        return self.projector.cand_embed.weight.shape[1]

    def get_loss(self, query_embeds, candidates, targets):
        """Compute loss.

        Args:
        - query_embeds: 2-D tensor of query embeddings, shape (batch
          size, dim)
        - candidates: 2-D tensor of candidate hypernym indices, shape
          (batch size, nb candidates)
        - targets: 2-D tensor of target labels (1 or 0), shape (batch
          size, nb candidates)

        Returns:
        - loss

        """
        logits = self._forward_to_logits(query_embeds, candidates)
        loss = self.loss_fn(logits, targets)
        return loss

    def _forward_to_logits(self, query_embeds, candidates):
        """Compute forward pass up to logits of candidate scores.

        Args:
        - query_embeds: 2-D tensor of query embeddings, shape (batch
          size, dim)
        - candidates: 2-D tensor of candidate indices, shape (batch
          size, nb candidates)

        Returns:
        - 2-D tensor of scores, shape (batch size, nb candidates per
          query

        """
        features = self.projector(query_embeds, candidates)
        logits = self.output(features.transpose(1,2)).squeeze(2)
        return logits

    def forward(self, query_embeds, candidates):
        """Compute forward pass and return candidate scores.

        Args:
        - query_embeds: 2-D tensor of query embeddings, shape (batch
          size, dim)
        - candidates: 2-D tensor of candidate indices, shape (batch
          size, nb candidates)

        Returns:
        - 2-D tensor of scores, shape (batch size, nb candidates per
          query

        """
        logits = self._forward_to_logits(query_embeds, candidates)
        return nn.sigmoid(logits.clamp(-10,10))


        
class Projector(nn.Module):
    """ Feature extractor which exploits projection learning. Computes
    the forward pass up to the dot product of the query projections
    and the candidates, which we will call features.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialize. 

        Args:
        - cand_embed: an Embedding object for the candidate hypernyms
        - nb_proj: nb of feature maps
        - dropout: probability of dropout
        - normalize_e: normalize embeddings after lookup
        - normalize_p: normalize projections

        Keyword args:
        - cuda: boolean which specifies whether we use cuda
        - seed: seed for RNG

        """
        super(Projector, self).__init__()

        # Process keyword args
        if "seed" in kwargs and kwargs["seed"]:
            torch.manual_seed(kwargs["seed"])
        if "cuda" not in kwargs:
            kwargs["cuda"] = False
        self.use_cuda = kwargs["cuda"]
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        # Process positional args.
        if len(args) != 5:
            msg = "Expected 0 or 5 positional args, found {}.".format(len(args))
            raise TypeError(msg)
        self.cand_embed = args[0]
        self.dim = self.cand_embed.weight.shape[1]
        self.nb_proj = args[1]
        dropout = args[2]
        self.normalize_e = args[3]
        self.normalize_p = args[4]
        self.dropout = nn.Dropout(p=dropout)
        # Initialize projection matrices using scheme from Glorot &
        # Bengio (2008).
        var = 2 / (self.dim + self.dim)
        mat_data = torch.zeros([self.nb_proj, self.dim, self.dim], dtype=torch.float32, device=device)
        mat_data.normal_(0, var)
        mat_data += torch.cat([torch.eye(self.dim, device=device).unsqueeze(0) for _ in range(self.nb_proj)])
        self.pmats = nn.Parameter(mat_data, device=device)

    def get_nb_projections(self):
        return self.nb_proj


    def _get_projections(self, embeds):
        """ Get projections of embeddings.

        Args:
        - embeds: 2-D tensor of embeddings, shape (batch size,
          dim)

        Returns:
        - Return 3-D tensor of projections, (batch size, nb
          projections, dim)

        """
        
        if self.normalize_e:
            embeds = F.normalize(embeds, p=2, dim=1)
        if self.training:
            embeds = self.dropout(embeds)
        p = torch.matmul(self.pmats, embeds.transpose(0,1))
        if self.normalize_p:
            proj = F.normalize(proj, p=2, dim=1)
        if self.training:
            p = self.dropout(p)
        p = p.transpose(0,1).transpose(0,2)
        return p

    def _get_features(self, query_embeds, candidates):
        """
        Get features (compute forward pass up to the dot product
        of the query projections and candidate embeddings).

        Args:
        - query_embeds: 2-D tensor of query embeddings, shape (batch
          size, dim)
        - candidates: 2-D tensor of candidate indices, shape (batch
          size, nb candidates per query)

        Returns:
        - 3-D tensor of features, shape (batch size, nb projections,
          nb candidates per query).

        """
        p_q = self._get_projections(query_embeds)
        h_embeds = self.cand_embed(candidates)
        if self.normalize_e:
            h_embeds = F.normalize(h_embeds, p=2, dim=2)
        if self.training:
            h_embeds = self.dropout(h_embeds)
        h_embeds = h_embeds.transpose(1,2)
        features = torch.bmm(p_q, h_embeds)
        return features

    def forward(self, query_embeds, candidates):
        """
        Compute features of (query, candidate) pairs.

        Args:
        - query_embeds: 2-D tensor of query embeddings, shape (batch
          size, dim)
        - candidates: 2-D tensor of candidate indices, shape (batch
          size, nb candidates per query)

        Returns:
        - 3-D tensor of features, shape (batch size, nb projections,
          nb candidates per query)

        """
        return self._get_features(query_embeds, candidates)
