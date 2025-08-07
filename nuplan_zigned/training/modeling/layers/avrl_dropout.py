from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class MyDropout(nn.modules.dropout._DropoutNd):
    def __init__(self,
                 p: float = 0.5,
                 inplace: bool = False,
                 bools: Dict[str, bool] = {'store_dropout_mask': False,
                                           'use_frozen_dropout_mask': False}) -> None:
        super(MyDropout, self).__init__(p=p, inplace=inplace)
        self.bools = bools
        self.frozen_dropout_mask = None

    def forward(self, input: Tensor) -> Tensor:  # all batches within a single call share the same dropout mask
        if self.training:
            batch_size, hidden_dim = input.size()

            # # old version
            # dropout_mask = torch.bernoulli((1 - self.p) * torch.ones((1, hidden_dim), device=input.device))
            # dropout_mask = dropout_mask.repeat_interleave(batch_size, dim=0)

            # efficient version
            if self.bools['use_frozen_dropout_mask']:
                dropout_mask = self.frozen_dropout_mask
            else:
                dropout_mask = torch.bernoulli((1 - self.p) * torch.ones((1, hidden_dim), device=input.device))
            if self.bools['store_dropout_mask']:
                self.frozen_dropout_mask = dropout_mask

            # dropout_mask = dropout_mask.repeat_interleave(batch_size, dim=0)  # this is slow because it copies data
            dropout_mask = dropout_mask.expand(batch_size, -1)  # Returns a new view of the self tensor with singleton dimensions expanded to a larger size.

            return input * dropout_mask
        else:
            return input