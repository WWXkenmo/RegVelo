import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


class GRN_layer(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""
    def __init__(self, n_targets, n_regulators, W_init, prior_net,lamba = 1.,bias=False, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                  local_rep=False, **kwargs):
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
        self.prior_prec = weight_decay
        self.weights = Parameter(W_init)
        self.prior_net = prior_net.float()
        self._size = self.weights.size()
        self.qz_loga = Parameter(torch.Tensor(self._size))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.bias = None
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        self.weights.data = self.weights.data * self.prior_net
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw_col = (.5 * self.prior_prec * self.weights.pow(2)) + self.lamba
        logpw = torch.sum((1 - q0) * logpw_col)
        logpb = 0 
        return logpw + logpb

    
    def _get_penalty(self):
        return self._reg_w()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self._size)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask * self.weights

    def forward(self, input):
        weights = self.sample_weights()
        output = input * weights
        
        return output

    def _get_mask(self):
        return F.sigmoid(self.qz_loga) * (limit_b - limit_a) + limit_a

    def _get_grn(self):
        mask = self._get_mask()
        weights = self.weights * mask

        return weights

    def __repr__(self):
        s = ('{name}(Gene Regulatory Network Layer, n_regulators={n_regulators}, n_targets={n_targets}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
            )
        s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)