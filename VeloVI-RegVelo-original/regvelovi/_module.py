"""Main module."""
from typing import Callable, Iterable, Literal, Optional, Any

import numpy as np
import torch
import math
import gpytorch
import torchdiffeq
from torchdiffeq import odeint
import torch.nn.functional as F
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl
import torchode as to
from ._constants import REGISTRY_KEYS

torch.backends.cudnn.benchmark = True


class DecoderVELOVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    linear_decoder
        Whether to use linear decoder for time
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        velocity_mode: Literal["global","decouple"] = "global",
        **kwargs,
    ):
        super().__init__()
        self.n_ouput = n_output
        self.linear_decoder = linear_decoder
        self.mode = velocity_mode

        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        if self.mode == "decouple":
            self.pi_first_decoder = FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                **kwargs,
            )

            # categorical pi
            # 3 states
            # induction, repression, repression steady state
            self.px_pi_decoder = nn.Linear(n_hidden, 3 * n_output)

            # rho for induction
            self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())
            
            # tau for repression
            self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

            self.linear_scaling_tau = nn.Parameter(torch.zeros(n_output))
            self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(n_output))

        if self.mode == "global":
            self.global_t_decoder = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())

    def forward(self, z: torch.Tensor, latent_dim: int = None):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask
        rho_first = self.rho_first_decoder(z_in)

        if self.mode == "decouple":
            # The decoder returns values for the parameters of the ZINB distribution
            if not self.linear_decoder:
                px_rho = self.px_rho_decoder(rho_first)
                px_tau = self.px_tau_decoder(rho_first)
            else:
                px_rho = nn.Sigmoid()(rho_first)
                px_tau = 1 - nn.Sigmoid()(
                    rho_first * self.linear_scaling_tau.exp()
                    + self.linear_scaling_tau_intercept
                )

            # cells by genes by 3
            pi_first = self.pi_first_decoder(z)
            px_pi = nn.Softplus()(
                torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 3))
            )

            return px_pi, px_rho, px_tau
        
        if self.mode == "global":
            prob_t = self.global_t_decoder(rho_first)

            return prob_t

## define gpytorch gaussian process emulator
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    """ 
    Interpolate the regulator use the multitask variational gaussian process
    stochastic training, feasible to minibatch training (not like exact inference)
    need to define following parameters

    m_induce_point: induce point number
    num_latents: the number of latent functions
    output_dim: number of the tasks (output dimension e.g. number of regulators+CREs)
    """

    def __init__(
        self,
        output_dim: int = None,
        grid_bounds = (0.,20.),
        num_latents: int = 10,
        grid_size: int = 64,
        ):
        # Let's use a different set of inducing points for each latent function
        #inducing_points = torch.rand(num_latents, m_induce_point, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points = grid_size, batch_shape=torch.Size([output_dim])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # TODO: use GridInterpolationVariationalStrategy to as base variational strategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size = grid_size, grid_bounds=[grid_bounds],
                variational_distribution = variational_distribution,
            ),
            num_tasks=output_dim,
            #num_latents=num_latents,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([output_dim])),
            batch_shape=torch.Size([output_dim])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

## define a new class velocity encoder
class velocity_encoder(nn.Module):
    """ 
    encode the velocity
    time dependent transcription rate is determined by upstream emulator
    velocity could be build on top of this

    parameters:
    n_tf: the number of upstream regulator (transcription factors)
    n_cre: the number of CRE (cis-regulatory elements)
    interpolator: method to perform interpolation: "GP" or "RBF"
    W: skeleton matrix (base GRN, two types, targets*regulators, targets*(regulator*CRE))
    W_int: the initialisation of GRN
    log_h_int: the initialisation of log_h (transcription factors)
    log_h_cre_int: the initialisation of log_h (CRE)

    merge velocity encoder and emulator class
    """                 
    def __init__(
        self,
        n_tf: int = 5,
        n_cre: int = None,
        interpolator: Literal["GP", "RBF"] = "GP",
        velocity_mode: Literal["global","decouple"] = "global",
        pair_indices: torch.Tensor = None,
        W: torch.Tensor = (torch.FloatTensor(5, 5).uniform_() > 0.5).int(),
        W_int: torch.Tensor = None,
        log_h_int: torch.Tensor = None,
        log_h_cre_int: torch.Tensor = None,
    ):
        device = W.device
        super().__init__()
        self.n_tf = n_tf
        self.n_cre = n_cre
        self.mode = velocity_mode
        self.interpolator = interpolator
        self.pair_indices = pair_indices

        if velocity_mode == "global":
            if interpolator == "GP":
                if self.pair_indices is not None:
                    self.multitaskGP_CRE = MultitaskGPModel(output_dim = n_cre, num_latents = 20).to(device)
                    self.likelihoodGP_CRE = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_cre).to(device)
                    ## TODO: add device
                ## add TF expression
                self.multitaskGP_TF = MultitaskGPModel(output_dim = n_tf, num_latents = 20).to(device)
                self.likelihoodGP_TF = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tf).to(device)

            if interpolator == "RBF":
                if self.pair_indices is not None:
                    self.log_h_cre = torch.nn.Parameter(log_h_cre_int)
                    self.log_phi_cre = torch.nn.Parameter(torch.ones(n_cre).to(device))
                    self.tau_cre = torch.nn.Parameter(torch.ones(n_cre).to(device)*7)
                    self.o_cre = torch.nn.Parameter(torch.ones(n_cre).to(device))

                self.log_h = torch.nn.Parameter(log_h_int)
                self.log_phi = torch.nn.Parameter(torch.ones(n_tf).to(device))
                self.tau = torch.nn.Parameter(torch.ones(n_tf).to(device)*10)
                self.o = torch.nn.Parameter(torch.ones(n_tf).to(device))

        if velocity_mode == "decouple":
            self.log_h = torch.nn.Parameter(log_h_int.repeat(W.shape[0],1)*W)
            self.log_phi = torch.nn.Parameter(torch.ones(W.shape).to(device)*W)
            self.tau = torch.nn.Parameter(torch.ones(W.shape).to(device)*W*7)
            self.o = torch.nn.Parameter(torch.ones(W.shape).to(device)*W)

        self.mask_m = W
        ## initialize grn
        self.grn = torch.nn.Parameter(W_int*self.mask_m)

        ## initilize gamma and beta
        #self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_int))
        #self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_int))
        #self.alpha_unconstr_max = torch.nn.Parameter(torch.randn(n_int))
        #self.alpha_unconstr = torch.nn.Parameter(torch.tensor(alpha_unconstr_init).to(device))
        
        ## initialize logic gate multivariate bernoulli distribution
        ## using hook to mask the gradients
        ### define hook to froze the parameters

    def _set_mask_grad(self):
        self.hooks_grn = []
        #mask_m = self.mask_m
        
        def _hook_mask_no_regulator(grad):
            return grad * self.mask_m
        
        if self.mode == "decouple":
            self.hooks_log_h = []
            self.hooks_log_phi = []
            self.hooks_tau = []
            self.hooks_o = []

            w_log_h = self.log_h.register_hook(_hook_mask_no_regulator)
            w_log_phi = self.log_phi.register_hook(_hook_mask_no_regulator)
            w_tau = self.tau.register_hook(_hook_mask_no_regulator)
            w_o = self.o.register_hook(_hook_mask_no_regulator)

            self.hooks_log_h.append(w_log_h)
            self.hooks_log_phi.append(w_log_phi)
            self.hooks_tau.append(w_tau)
            self.hooks_o.append(w_o)

        ## frozen the weight
        w_grn = self.grn.register_hook(_hook_mask_no_regulator)
        self.hooks_grn.append(w_grn)

    def emulation_all(self,t: torch.Tensor = None):

        if self.mode == "decouple":
            emulate_m = []
            h = torch.exp(self.log_h)
            phi = torch.exp(self.log_phi)
            for i in range(t.shape[1]):
                # for each time stamps, predict the emulator predict value
                tt = t[:,i]
                emu = h * torch.exp(-phi*(tt.reshape((len(tt),1))-self.tau)**2) + self.o
                emu = emu * self.mask_m
                emulate_m.append(emu)
            
            emulate_m = torch.stack(emulate_m,2)
            emulate_CRE_m = None

        if self.mode == "global":
            if self.interpolator == "GP":
                if self.pair_indices is not None:
                    emulate_CRE_m = self.multitaskGP_CRE(t)
                else:
                    emulate_CRE_m = None
                emulate_m = self.multitaskGP_TF(t)

            if self.interpolator == "RBF":
                if self.pair_indices is not None:
                    emulate_CRE_m = []
                    h = torch.exp(self.log_h_cre)
                    phi = torch.exp(self.log_phi_cre)
                    for i in range(len(t)):
                        # for each time stamps, predict the emulator predict value
                        tt = t[i]
                        emu = h * torch.exp(-phi*(tt-self.tau_cre)**2) + self.o_cre
                        emulate_CRE_m.append(emu)

                    emulate_CRE_m = torch.stack(emulate_CRE_m,1)
                else:
                    emulate_CRE_m = None

                emulate_m = []
                h = torch.exp(self.log_h)
                phi = torch.exp(self.log_phi)
                for i in range(len(t)):
                    # for each time stamps, predict the emulator predict value
                    tt = t[i]
                    emu = h * torch.exp(-phi*(tt-self.tau)**2) + self.o
                    emulate_m.append(emu)

                emulate_m = torch.stack(emulate_m,1)

        return emulate_m, emulate_CRE_m

    ## TODO: introduce sparsity in the model
    def forward(self,t,y):
        ## split x into unspliced and spliced readout
        ## x is a matrix with (G*2), in which row is a subgraph (batch)
        #print(y)
        #assert torch.jit.isinstance(args, torch.Tensor)
        #assert torch.jit.isinstance(t, torch.Tensor)
        #assert torch.jit.isinstance(y, torch.Tensor)
        if self.mode == "global":
            if y.shape[0] == 1:
                y = y.ravel()
            n_int = len(y)
            if len(y.shape) == 1:
                u = y[0:int(n_int/2)]
                s = y[int(n_int/2):n_int]
            else:
                u = y[:,0:int(n_int/2)]
                s = y[:,int(n_int/2):n_int]
        else:
            if len(y.shape) == 1:
                u = y[0]
                s = y[1]
            else:  
                u = y[:,0]
                s = y[:,1]

        if self.mode == "global":
            if self.interpolator == "GP":
                emu_tf  = self.likelihoodGP_TF(self.multitaskGP_TF(t.view(-1))).mean
                if self.TF_func is not None:
                    emu_tf = torch.matmul(emu_tf,self.TF_func[:,:,1].T)
                if self.pair_indices is not None:
                    ## calculate transcription activate rate use CRE and TF
                    emu_cre = self.likelihoodGP_CRE(self.multitaskGP_CRE(t.view(-1))).mean
                    if self.CRE_func is not None:
                        emu_cre = torch.matmul(emu_cre,self.CRE_func[:,:,1].T)
                    emu_cre_tf = emu_tf[:,self.pair_indices[:,0]] * emu_cre[:,self.pair_indices[:,1]]
                    alpha_unconstr = torch.matmul(emu_cre_tf,self.grn.T)
                else:
                    alpha_unconstr = torch.matmul(emu_tf,self.grn.T)
            
            if self.interpolator == "RBF":
                h = torch.exp(self.log_h)
                phi = torch.exp(self.log_phi)
                #emu = h[locate,:] * torch.exp(-phi[locate,:]*(T.reshape((dim,1))-self.tau[locate,:])**2) + self.o[locate,:]
                emu_tf = h * torch.exp(-phi*(t - self.tau)**2) + self.o
                if self.pair_indices is not None:
                    h = torch.exp(self.log_h_cre)
                    phi = torch.exp(self.log_phi_cre)
                    emu_cre = h * torch.exp(-phi*(t - self.tau_cre)**2) + self.o_cre
                    emu_cre_tf = emu_tf[self.pair_indices[:,0]] * emu_cre[self.pair_indices[:,1]]
                    alpha_unconstr = torch.matmul(emu_cre_tf.unsqueeze(0),self.grn.T)
                else:
                    ## Use the Emulator matrix to predict alpha
                    alpha_unconstr = torch.matmul(emu_tf.unsqueeze(0),self.grn.T)
                
            #alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias[locate]
            alpha_unconstr = alpha_unconstr.ravel()
            alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias

            ## Generate kinetic rate
            #beta = torch.clamp(F.softplus(self.beta_mean_unconstr[locate]), 0, 50)
            #gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr[locate]), 0, 50)
            beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
            gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
            alpha = torch.clamp(alpha_unconstr,1e-3,) ## set a minimum transcriptional rate to prevent it can't be trained
            alpha = F.softsign(alpha)*torch.clamp(F.softplus(self.alpha_unconstr_max), 0, 50)
            #alpha = F.softplus(alpha_unconstr)                      
        
        if self.mode == "decouple":
            ## calculate emulator value
            ## t is a vector per gene (G*1)
            ## extract the corresponding gene
            #u = u[locate]
            #s = s[locate]
            #T = t[locate]
            
            if self.interpolator == "RBF":
                h = torch.exp(self.log_h)
                phi = torch.exp(self.log_phi)
                #emu = h[locate,:] * torch.exp(-phi[locate,:]*(T.reshape((dim,1))-self.tau[locate,:])**2) + self.o[locate,:]
                emu = h * torch.exp(-phi*(t.reshape((-1,1)) - self.tau)**2) + self.o

                ## Use the Emulator matrix to predict alpha
                #emu = emu * self.grn[locate,:]
                emu = emu * self.grn
                alpha_unconstr = emu.sum(dim=1)

            #alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias[locate]
            alpha_unconstr = alpha_unconstr + self.alpha_unconstr_bias

            ## Generate kinetic rate
            #beta = torch.clamp(F.softplus(self.beta_mean_unconstr[locate]), 0, 50)
            #gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr[locate]), 0, 50)
            beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
            gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
            alpha = torch.clamp(alpha_unconstr,1e-3,) ## set a minimum transcriptional rate to prevent it can't be trained
            alpha = F.softsign(alpha_unconstr)*torch.clamp(F.softplus(self.alpha_unconstr_max), 0, 50)
            #alpha = F.softplus(alpha_unconstr)

        ## Predict velocity
        du = alpha - beta*u
        ds = beta*u - gamma*s
        
        du = du.reshape((-1,1))
        ds = ds.reshape((-1,1))

        v = torch.concatenate([du,ds],axis = 1)

        if len(y.shape) == 1:
            v = v.view(-1)
        
        if self.mode == "global":
            v = torch.cat((du.ravel(),ds.ravel()))
            v = v.unsqueeze(0)

        return v
    
# VAE model
class VELOVAE(BaseModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the veloVI model descibed in :cite:p:`GayosoWeiler2022`

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    alpha_1 represent the maximum transcription rate once could reach in induction stage
    """

    def __init__(
        self,
        n_input: int,
        regulator_index,
        target_index,
        skeleton,
        corr_m,
        readout_type,
        tf_cre_pair: Optional[np.ndarray] = None,
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        lam: float = 1.,
        num_trainset: int = 1000,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: Optional[np.ndarray] = None,
        alpha_unconstr_init: Optional[np.ndarray] = None,
        alpha_1_unconstr_init: Optional[np.ndarray] = None,
        log_h_int: Optional[np.ndarray] = None,
        log_h_cre_int: Optional[np.ndarray] = None,
        switch_spliced: Optional[np.ndarray] = None,
        switch_unspliced: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 1/3,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = True,
        velocity_mode: Literal["global","decouple"] = "global",
        interpolator: Literal["RBF", "GP"] = "GP",
        TF_func: Optional[torch.Tensor] = None,
        CRE_func: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate
        self.interpolator = interpolator
        self.mode = velocity_mode

        self.lamba = lam
        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input
        n_targets = sum(target_index)
        n_regulators = sum(regulator_index)
        if tf_cre_pair is not None:
            n_cre = sum(readout_type == "accessibility")
        
        self.n_targets = int(n_targets) 
        self.n_regulators = int(n_regulators)
        self.regulator_index = regulator_index
        self.target_index = target_index
        self.readout_type = readout_type
        
        if velocity_mode == "decouple":
            # switching time for each target gene
            self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_targets))
            if true_time_switch is not None:
                self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
            else:
                self.true_time_switch = None

        # TODO: Add `require_grad`
        ## The maximum transcription rate (alpha_1) for each target gene 
        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = torch.nn.Parameter(torch.ones(n_targets))
        else:
            self.alpha_1_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_1_unconstr_init)
            )
            self.alpha_1_unconstr.data = self.alpha_1_unconstr.data.float()

        # degradation for each target gene
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_targets))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(gamma_unconstr_init)
            )

        # splicing for each target gene
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_targets))

        # transcription (bias term for target gene transcription rate function)
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_targets))
        else:
            self.alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(alpha_unconstr_init)
            )

        # likelihood dispersion
        # for now, with normal dist, this is just the variance for target genes
        var_num = 3
        if velocity_mode == "global":
            var_num = 1

        self.scale_unconstr_targets = torch.nn.Parameter(-1 * torch.ones(n_targets*2, var_num))
        
        ## TODO: use normal dist to model the emulator preduction
        #self.scale_unconstr_regulators = torch.nn.Parameter(-1 * torch.ones(n_regulators, 1))
        """
        need discussion
        self.scale_unconstr_regulators = torch.nn.Parameter(-1 * torch.ones(n_regulators*2, 3))
        """

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_target-d data
        n_input_decoder = n_latent
        n_output_decoder = n_targets
        
        if velocity_mode == "global":
            n_output_decoder = 1

        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_output_decoder,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
            velocity_mode = velocity_mode,
        )
        
        # define velocity encoder, define velocity vector for target genes
        if TF_func is not None:
            n_regulators = TF_func.shape[1]
        if CRE_func is not None:
            n_cre = CRE_func.shape[1]
        self.v_encoder = velocity_encoder(n_tf = n_regulators, 
                                          n_cre = n_cre if tf_cre_pair is not None else None,
                                          interpolator = interpolator,
                                          velocity_mode = velocity_mode,
                                          pair_indices = tf_cre_pair if tf_cre_pair is not None else None,
                                          W = skeleton, W_int = corr_m, log_h_int= log_h_int,
                                          log_h_cre_int = log_h_cre_int if log_h_cre_int is not None else None,)
        #require parameter for intergral
        self.x0 = torch.zeros(self.n_targets*2).unsqueeze(0).to(self.device)
        self.dt0 = torch.full((self.x0.shape[0],), 1)
        
        # saved kinetic parameter in velocity encoder module
        ## load require parameter for output velocity
        self.v_encoder.TF_func = TF_func
        self.v_encoder.CRE_func = CRE_func

        self.v_encoder.beta_mean_unconstr = self.beta_mean_unconstr
        self.v_encoder.gamma_mean_unconstr = self.gamma_mean_unconstr
        self.v_encoder.alpha_unconstr_max = self.alpha_1_unconstr
        self.v_encoder.alpha_unconstr_bias = self.alpha_unconstr
        # initilize grn (masked parameters)
        self.v_encoder._set_mask_grad()

        # define mll loss for GP
        if interpolator == "GP":
            self.v_encoder.mll_TF = gpytorch.mlls.PredictiveLogLikelihood(self.v_encoder.likelihoodGP_TF, self.v_encoder.multitaskGP_TF, num_data=num_trainset, beta = 0.1)
            if tf_cre_pair is not None:
                self.v_encoder.mll_CRE = gpytorch.mlls.PredictiveLogLikelihood(self.v_encoder.likelihoodGP_CRE, self.v_encoder.multitaskGP_CRE, num_data=num_trainset, beta = 0.1)
            
    def _get_inference_input(self, tensors):
        readouts = tensors[REGISTRY_KEYS.X_KEY]
        spliced = readouts[:,self.readout_type == "spliced"]
        unspliced = readouts[:,self.readout_type == "unspliced"]

        if 'accessibility' in list(set(self.readout_type)):
            accessibility = readouts[:,self.readout_type == "accessibility"]
        else:
            accessibility = None
        #regulator_spliced = spliced[:,self.regulator_index]
        #target_spliced = spliced[:,self.target_index,:]
        #target_unspliced = unspliced[:,self.target_index,:]
        
        input_dict = {
            "spliced": spliced,
            "unspliced": unspliced,
            "accessibility": accessibility,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "beta": beta,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced,
        unspliced,
        accessibility,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        if accessibility is not None:
            encoder_input = torch.cat((spliced_, unspliced_, accessibility), dim=-1)
        else:
            encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, beta = self._get_rates()

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "gamma": gamma,
            "beta": beta,
        }
        return outputs

    def _get_rates(self):
        # globals
        # degradation for each target gene
        gamma = torch.clamp(F.softplus(self.v_encoder.gamma_mean_unconstr), 0, 50)
        # splicing for each target gene
        beta = torch.clamp(F.softplus(self.v_encoder.beta_mean_unconstr), 0, 50)
        # transcription for each target gene (bias term)
        #alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)

        return gamma, beta

    @auto_move_data
    def generative(self, z, gamma, beta, latent_dim=None):
        """Runs the generative model."""
        decoder_input = z

        if self.mode == "decouple":
            px_pi_alpha, px_rho, px_tau = self.decoder(decoder_input, latent_dim=latent_dim)
            px_pi = Dirichlet(px_pi_alpha).rsample()

            scale_unconstr = self.scale_unconstr_targets
            scale = F.softplus(scale_unconstr)

            ####################
            #scale_unconstr_regulators = self.scale_unconstr_regulators
            #scale_regulators = F.softplus(scale_unconstr_regulators)
            ####################

            mixture_dist_s, mixture_dist_u, emulation, emulation_cre = self.get_px(
                px_pi,
                px_rho,
                px_tau,
                scale,
                gamma,
                beta,
            )

            return {
                "px_pi": px_pi,
                "px_rho": px_rho,
                "px_tau": px_tau,
                "scale": scale,
                "px_pi_alpha": px_pi_alpha,
                "mixture_dist_u": mixture_dist_u,
                "mixture_dist_s": mixture_dist_s,
                "emulation": emulation,
                "emulation_cre": emulation_cre,
            }
        
        if self.mode == "global":
            prob_t = self.decoder(decoder_input, latent_dim = latent_dim)
            
            scale_unconstr = self.scale_unconstr_targets
            scale = F.softplus(scale_unconstr)

            mixture_dist_s, mixture_dist_u, emulation, emulation_cre = self.get_px_global(
                prob_t,
                scale,
                gamma,
                beta
            )

            return {
                "prob_t": prob_t,
                "scale": scale,
                "mixture_dist_u": mixture_dist_u,
                "mixture_dist_s": mixture_dist_s,
                "emulation": emulation,
                "emulation_cre": emulation_cre
            }
            
    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        readouts = tensors[REGISTRY_KEYS.X_KEY]
        spliced = readouts[:,self.readout_type == "spliced"]
        unspliced = readouts[:,self.readout_type == "unspliced"]

        if 'accessibility' in list(set(self.readout_type)):
            accessibility = readouts[:,self.readout_type == "accessibility"]
        else:
            accessibility = None

        ### extract spliced, unspliced readout
        regulator_spliced = spliced[:,self.regulator_index]
        target_spliced = spliced[:,self.target_index]
        target_unspliced = unspliced[:,self.target_index]
        
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        ## state parameter only would be used when use decouple mode
        if self.mode == "decouple":
            px_pi = generative_outputs["px_pi"]
            px_pi_alpha = generative_outputs["px_pi_alpha"]

        #end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        ##################
        # dist_emulate = generative_outputs["dist_emulate"]
        ##################

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -mixture_dist_s.log_prob(target_spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(target_unspliced)
        
        ############################
        #reconst_loss_regulator = -dist_emulate.log_prob(regulator_spliced)
        #reconst_loss_regulator = reconst_loss_regulator.sum(dim=-1)
        ############################

        reconst_loss_target = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)
        
        ### calculate the reconstruct loss for emulation
        emulation = generative_outputs["emulation"]
        emulation_cre = generative_outputs["emulation_cre"]
        if self.interpolator == "RBF":
            recon_loss_reg = F.mse_loss(regulator_spliced, emulation, reduction='none').sum(-1)
            if emulation_cre is not None:
                recon_loss_reg = (F.mse_loss(regulator_spliced, emulation, reduction='none').sum(-1) + F.mse_loss(accessibility, emulation_cre, reduction='none').sum(-1))/2
        if self.interpolator == "GP":
            if self.v_encoder.TF_func is not None:
                regulator_spliced = torch.matmul(regulator_spliced,self.v_encoder.TF_func[:,:,0])
            recon_loss_reg = -self.v_encoder.mll_TF(emulation,regulator_spliced)
            #print(self.v_encoder.likelihoodGP_TF(emulation).mean.shape)
            #recon_loss_reg = F.mse_loss(regulator_spliced, self.v_encoder.likelihoodGP_TF(emulation).mean, reduction='none').sum(-1)
            if emulation_cre is not None:
                if self.v_encoder.CRE_func is not None:
                    accessibility = torch.matmul(accessibility,self.v_encoder.CRE_func[:,:,0])
                recon_loss_reg = -0.5*(self.v_encoder.mll_TF(emulation,regulator_spliced).item()+self.v_encoder.mll_CRE(emulation_cre,accessibility).item())

        if self.mode == "decouple":
            kl_pi = kl(
                Dirichlet(px_pi_alpha),
                Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
            ).sum(dim=-1)
        else:
            kl_pi = 0

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        if self.interpolator == "RBF":
            local_loss = torch.mean(reconst_loss_target + recon_loss_reg + weighted_kl_local)
        if self.interpolator == "GP":
            local_loss = torch.mean(reconst_loss_target + weighted_kl_local) + recon_loss_reg

        # recon_loss_all = local_loss - kl_local
        # add L1 loss to grn
        #L1_loss = torch.abs(self.v_encoder.grn).mean()
        L1_loss = 0
        # add L2 loss to grn
        # L2_loss = torch.norm(self.v_encoder.grn.flatten(), 2)**2
        loss = local_loss + self.lamba * L1_loss
        recon_loss_reg = torch.tensor(recon_loss_reg).unsqueeze(0) 
        #print(recon_loss_reg)
        loss_recoder = LossOutput(
                loss=loss, reconstruction_loss=recon_loss_reg, kl_local=torch.tensor(0)
            )

        return loss_recoder

    @auto_move_data
    def get_px_global(
        self,
        prob_t,
        scale,
        gamma,
        beta,
    ) -> torch.Tensor:
        t = prob_t * self.t_max
        n_cells = len(prob_t)
        mean_u, mean_s = self._get_global_unspliced_spliced(
            t
        )

        emulation, emulation_cre = self.v_encoder.emulation_all(t)
        if self.interpolator == "RBF":
            emulation = emulation.T
            if emulation_cre is not None:
                emulation_cre = emulation_cre.T

        scale_u = scale[: self.n_targets, 0].expand(n_cells, self.n_targets).sqrt()
        scale_s = scale[self.n_targets :, 0].expand(n_cells, self.n_targets).sqrt()

        dist_u = Normal(mean_u, scale_u)
        dist_s = Normal(mean_s, scale_s)
        
        return dist_s, dist_u, emulation, emulation_cre

    @auto_move_data
    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
    ) -> torch.Tensor:
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # predict the abundance in induction phase for target genes
        ind_t = t_s * px_rho
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            ind_t
        )
        
        # get the emulation results of each regulator to the inductive latent time
        
        emulation, emulation_cre = self.v_encoder.emulation_all(ind_t.T)
        emulation = emulation.mean(dim=0).T
       
        
        #######################
        #scale_regulators = scale_regulators.expand(n_cells, self.n_regulators, 1).sqrt()
        #######################

        ### only have three cell state
        # induction
        # repression
        # repression steady state
        scale_u = scale[: self.n_targets, :].expand(n_cells, self.n_targets, 3).sqrt()

        # calculate the initial state for repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            t_s.reshape(1,len(t_s))
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)
        scale_s = scale[self.n_targets :, :].expand(n_cells, self.n_targets, 3).sqrt()

        #end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
        #    (s_0 - self.switch_spliced).pow(2)
        #).sum()

        # unspliced
        mean_u = torch.stack(
            (
                mean_u_ind,
                mean_u_rep,
                mean_u_rep_steady,
            ),
            dim=2,
        )
        scale_u = torch.stack(
            (
                scale_u[..., 0],
                scale_u[..., 0],
                0.1 * scale_u[..., 0],
            ),
            dim=2,
        )
        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        # spliced
        mean_s = torch.stack(
            (mean_s_ind, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )
        scale_s = torch.stack(
            (
                scale_s[..., 0],
                scale_s[..., 0],
                0.1 * scale_s[..., 0],
            ),
            dim=2,
        )
        dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        # emulation
        #dist_emulate = Normal(emulation.T,scale_regulators[...,0])
        #############

        return mixture_dist_s, mixture_dist_u, emulation, emulation_cre

    def _get_global_unspliced_spliced(
        self, t, eps=1e-6
    ):
        """
        this function aim to calculate the spliced and unspliced aundance of target genes
        """
        def get_step_size(step_size, t1, t2, t_size):
            if step_size is None:
                options = {}
            else:
                step_size = (t2 - t1)/t_size/step_size
                options = dict(step_size = step_size)
            return options

        device = self.device
        t = t.ravel()
        if len(t)>1:
            t_eval, index = torch.sort(t)
            index2 = (t_eval[:-1] != t_eval[1:])
            #subtraction_values = torch.where((t_eval[1:] - t_eval[:-1])>0, (t_eval[1:] - t_eval[:-1]), torch.inf).min()
            #subtraction_values[subtraction_values == float("Inf")] = 0

            true_tensor = torch.ones(1, dtype=torch.bool)
            index2 = torch.cat((index2,true_tensor.to(index2.device)))

            #print(t_eval)
            t_eval[index2 == False] = t_eval[index2 == False] + 0.001
            t_eval, _ = torch.sort(t_eval)
        else:
            t_eval = t
        
        t_eval = torch.cat((torch.tensor([0]).to(index2.device),t_eval))
        #x0 = torch.zeros(self.n_targets*2).to(self.device)
        #x0 = x0.unsqueeze(0)
        #options = get_step_size(, t_eval[0], t_eval[-1], len(t_eval))
        #print(t_eval)
        #pred_x = odeint(self.v_encoder, x0, t_eval, method = 'dopri5').view(-1, self.n_targets*2)
        ## using torchode
        term = to.ODETerm(self.v_encoder)
        step_method = to.Dopri5(term = term)
        step_size_controller = to.FixedStepController()
        #dt0 = torch.full((x0.shape[0],), 1)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        sol = solver.solve(to.InitialValueProblem(y0=self.x0, t_eval=t_eval.repeat((1,1))), dt0=self.dt0)
        pred_x = sol.ys[0,:,:]
        pre_u = pred_x[1:,:self.n_targets]
        pre_s = pred_x[1:,self.n_targets:]
        
        if len(t)>1:
            unspliced = torch.zeros(pre_u.shape)
            spliced = torch.zeros(pre_s.shape)
            unspliced[index,:] = pre_u
            spliced[index,:] = pre_s

        if len(t)==0:
            unspliced = unspliced.ravel()
            spliced = spliced.ravel()
        
        return unspliced, spliced

        

    def _get_induction_unspliced_spliced(
        self, t, eps=1e-6
    ):
        """
        this function aim to calculate the spliced and unspliced abundance for target genes
        
        beta: the splicing parameter for each target gene
        gamma: the degradation parameter for each target gene
        
        ** the above parameters are saved in v_encoder
        t: target gene specific latent time
        """
        device = self.device
        t = t.T    
        
        if t.shape[1] > 1:
            t_eval, index = torch.sort(t, dim=1)
            index2 = (t_eval[:,:-1] != t_eval[:,1:])
            subtraction_values,_ = torch.where((t_eval[:,1:] - t_eval[:,:-1])>0, (t_eval[:,1:] - t_eval[:,:-1]), torch.inf).min(axis=1)
            subtraction_values[subtraction_values == float("Inf")] = 0
            
            true_tensor = torch.ones((t_eval.shape[0],1), dtype=torch.bool)
            index2 = torch.cat((index2, true_tensor.to(index2.device)),dim=1) ## index2 is used to get unique time points as odeint requires strictly increasing/decreasing time points
                
            subtraction_values = subtraction_values[None, :].repeat(index2.shape[1], 1).T
            t_eval[index2 == False] -= subtraction_values[index2 == False]*0.1
            ## extract initial target gene expression value
            #x0 = torch.cat((target_unspliced[:,0].reshape((target_unspliced.shape[0],1)),target_spliced[:,0].reshape((target_spliced.shape[0],1))),dim = 1)
            x0 = torch.zeros((t.shape[0],2)).to(self.device)
            #x0 = x0.double()
            t_eval = torch.cat((torch.zeros((t_eval.shape[0],1)).to(self.device),t_eval),dim=1)
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            
            term = to.ODETerm(self.v_encoder)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            dt0 = torch.full((x0.shape[0],), 1)
            sol = solver.solve(to.InitialValueProblem(y0=x0, t_eval=t_eval), dt0=dt0)
        else:
            t_eval = t
            t_eval = torch.cat((torch.zeros((t_eval.shape[0],1)).to(self.device),t_eval),dim=1)
            ## set up G batches, Each G represent a module (a target gene centerred regulon)
            ## infer the observe gene expression through ODE solver based on x0, t, and velocity_encoder
            x0 = torch.zeros((t.shape[0],2)).to(self.device)
            #x0 = x0.double()

            term = to.ODETerm(self.v_encoder)
            step_method = to.Dopri5(term=term)
            #step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
            step_size_controller = to.FixedStepController()
            solver = to.AutoDiffAdjoint(step_method, step_size_controller)
            #jit_solver = torch.jit.script(solver)
            dt0 = torch.full((x0.shape[0],), 1)
            sol = solver.solve(to.InitialValueProblem(y0=x0, t_eval=t_eval), dt0=dt0)

        ## generate predict results
        # the solved results are saved in sol.ys [the number of subsystems, time_stamps, [u,s]]
        pre_u = sol.ys[:,1:,0]
        pre_s = sol.ys[:,1:,1]     
        
        if t.shape[1] > 1:
            unspliced = torch.zeros_like(pre_u)
            spliced = torch.zeros_like(pre_s)   
            for i in range(index.size(0)):
                unspliced[i][index[i]] = pre_u[i]
                spliced[i][index[i]] = pre_s[i]
            unspliced = unspliced.T
            spliced = spliced.T
        else:
            unspliced = pre_u.ravel()
            spliced = pre_s.ravel()
    
        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (
            beta * u_0 / ((gamma - beta) + eps)
        ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
        return unspliced, spliced

    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights (for each Z, shape is genes by dim(Z)) in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        w = self.decoder.rho_first_decoder.fc_layers[0][0].weight
        if self.use_batch_norm_decoder:
            bn = self.decoder.rho_first_decoder.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = w
        loadings = loadings.detach().cpu().numpy()

        return loadings
