import math
import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))

class GRN_layer(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, n_targets, n_regulators, W_init, prior_net,loc_mean=0, loc_sdev=0.01, beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True, **kwargs):
        """
        :param n_targets: number of targets
        :param n_regulators: number of regulators
        :param W_init: GRN initialization
        :param prior_net: GRN Skeleton
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(GRN_layer, self).__init__()
        self.n_targets = n_targets
        self.n_regulators = n_regulators
        self.weights = Parameter(W_init*prior_net.float())
        self._size = self.weights.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

    def _get_mask(self):
        if self.training:
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
        else:
            s = F.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
        return hard_sigmoid(s)

    def _get_grn(self):
        mask = self._get_mask()
        weights = self.weights * mask

        return weights
        
    def _get_penalty(self):
        if self.training:
            penalty = F.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        else:
            penalty = 0
        
        return penalty

    def forward(self, input):
        mask = self._get_mask()
        weights = self.weights * mask
        output = input * weights
        return output

    def __repr__(self):
        s = ('{name}(Gene Regulatory Network Layer, n_regulators={n_regulators}, n_targets={n_targets}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
            )
        s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)





