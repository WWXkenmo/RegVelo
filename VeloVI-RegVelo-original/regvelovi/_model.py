import logging
import warnings
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import gpytorch
import torchdiffeq
from torchdiffeq import odeint
import torch.nn.functional as F
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.train import TrainingPlan, TrainRunner
from scvi.utils._docstrings import setup_anndata_dsp
from scvelo import logging as logg
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist

from ._constants import REGISTRY_KEYS
from ._module import VELOVAE
from ._regvelomixin import RegVeloMixin

logger = logging.getLogger(__name__)


def _softplus_inverse(x: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x)
    x_inv = torch.where(x > 20, x, x.expm1().log()).numpy()
    return x_inv


class REGVELOVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, RegVeloMixin):
    """Velocity Variational Inference.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~velovi.VELOVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    gamma_init_data
        Initialize gamma using the data-driven technique.
    linear_decoder
        Use a linear decoder from latent space to time.
    **model_kwargs
        Keyword args for :class:`~velovi.VELOVAE`
    """

    def __init__(
        self,
        adata: AnnData,
        W: torch.Tensor,
        corr_m: torch.Tensor = None,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_layers: int = 1,
        lam: float = 1.,
        dropout_rate: float = 0.1,
        gamma_init_data: bool = False,
        linear_decoder: bool = False,
        velocity_mode: Literal["global","decouple"] = "global",
        interpolator: Literal["RBF", "GP"] = "GP",
        SVD_transform: bool = True,
        k = 30,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.n_latent = n_latent

        readouts = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        spliced = readouts[:,adata.var["readout"] == "spliced"]
        unspliced = readouts[:,adata.var["readout"] == "unspliced"]

        if 'accessibility' in list(set(adata.var["readout"])):
            accessibility = readouts[:,adata.var["readout"] == "accessibility"]
        else:
            accessibility = None

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(int(spliced.shape[1]))][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(int(spliced.shape[1]))][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        alpha_unconstr = _softplus_inverse(us_upper)
        alpha_unconstr = np.asarray(alpha_unconstr).ravel()

        if interpolator == "RBF":
            log_h_int = torch.log(torch.max(torch.tensor(spliced), dim=0).values)
            if 'accessibility' in list(set(adata.var["readout"])):
                log_h_cre_int = torch.log(torch.max(torch.tensor(accessibility), dim=0).values)
            else:
                log_h_cre_int = None
        else:
            log_h_int = log_h_cre_int = None
        
        alpha_1_unconstr = np.zeros(us_upper.shape).ravel()
        #alpha_1_unconstr = _softplus_inverse(alpha_1_unconstr)
        
        if gamma_init_data:
            gamma_unconstr = np.clip(_softplus_inverse(us_upper / ms_upper), None, 10)
        else:
            gamma_unconstr = None

        ## initialization GRN attribute value (corr_m, regulator/target_index)
        if "regulators" not in adata.uns.keys() or "targets" not in adata.uns.keys():
            raise ValueError("adata need to save the names of regulators and targets")

        var = adata.var[:int(spliced.shape[1])]
        regulator_index = [i in adata.uns["regulators"] for i in var.index.values]
        target_index = [i in adata.uns["targets"] for i in var.index.values]
        
        if corr_m is None:
            if "pair_indices_atac" in list(adata.uns.keys()):
                reg = spliced[:,adata.uns["pair_indices_atac"][:,0]] * accessibility[:,adata.uns["pair_indices_atac"][:,1]]
                tar = spliced[:,target_index]
                corr_m = 1 - cdist(tar.T,reg.T, metric = 'correlation')
                corr_m = torch.tensor(corr_m)
            else:
                corr_m = 1 - cdist(spliced.T, spliced.T, metric='correlation')
                corr_m = torch.tensor(corr_m)
                corr_m = corr_m[target_index,]
                corr_m = corr_m[:,regulator_index]
        corr_m = corr_m.float()
        #corr_m = corr_m.to(self.device)
        
        ##
        if log_h_int is not None:
            log_h_int = log_h_int[regulator_index]
        alpha_1_unconstr = alpha_1_unconstr[target_index]
        alpha_unconstr = alpha_unconstr[target_index]

        ## use truncted SVD to reduce the dimension of X
        if SVD_transform:
            U, S, Vt = svds(spliced[:,regulator_index].T, k = k)
            ## define transformation matrix
            TF_EN = np.matmul(U,np.diag(1/S))
            TF_DE = np.matmul(U,np.diag(S))
            TF_EN = torch.tensor(TF_EN).unsqueeze(2)
            TF_DE = torch.tensor(TF_DE).unsqueeze(2)
            TF_func = torch.cat((TF_EN,TF_DE),axis = 2)
            if 'accessibility' in list(set(adata.var["readout"])):
                U, S, Vt = svds(accessibility.T, k = k)
                CRE_EN = np.matmul(U,np.diag(1/S))
                CRE_DE = np.matmul(U,np.diag(S))
                CRE_EN = torch.tensor(CRE_EN).unsqueeze(2)
                CRE_DE = torch.tensor(CRE_DE).unsqueeze(2)
                CRE_func = torch.cat((CRE_EN,CRE_DE),axis = 2)
            else:
                CRE_func = None
        else:
            CRE_func = TF_func = None

        n_input = adata.n_vars
        self.module = VELOVAE(
            n_input= n_input,
            regulator_index = regulator_index,
            target_index = target_index,
            skeleton = W,
            corr_m = corr_m,
            readout_type = adata.var["readout"],
            tf_cre_pair = adata.uns["pair_indices_atac"] if "pair_indices_atac" in list(adata.uns.keys()) else None,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            lam = lam,
            num_trainset = adata.obs.shape[0],
            dropout_rate=dropout_rate,
            gamma_unconstr_init=gamma_unconstr,
            alpha_unconstr_init=alpha_unconstr,
            alpha_1_unconstr_init=alpha_1_unconstr,
            log_h_int = log_h_int,
            log_h_cre_int = log_h_cre_int,
            linear_decoder=linear_decoder,
            velocity_mode = velocity_mode,
            interpolator = interpolator,
            TF_func = TF_func,
            CRE_func = CRE_func,
            **model_kwargs,
        )
        self._model_summary_string = (
            "REGVELOVI Model with the following params: \ninterpolator: {}, velocity_mode: {}, use_atac: {}, \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            interpolator,
            velocity_mode,
            str("pair_indices_atac" in list(adata.uns.keys())),
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())
        self.model_stage = "pretraining"

    def train(
        self,
        max_epochs: Optional[int] = 500,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        eps: float = 1e-16,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = True,
        gradient_clip_val: float = 10,
        plan_kwargs: Optional[dict] = None,
        optimizer: str = "AdamW",
        **trainer_kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        lr
            Learning rate for optimization
        weight_decay
            Weight decay for optimization
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        gradient_clip_val
            Val for gradient clipping
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        user_plan_kwargs = plan_kwargs.copy() if isinstance(plan_kwargs, dict) else {}
        plan_kwargs = {"lr": lr, "weight_decay": weight_decay, "optimizer": optimizer, "eps": eps}
        plan_kwargs.update(user_plan_kwargs)

        user_train_kwargs = trainer_kwargs.copy()
        trainer_kwargs = {"gradient_clip_val": gradient_clip_val}
        trainer_kwargs.update(user_train_kwargs)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        # trainer_kwargs["early_stopping_patience"] = 45
        if self.model_stage == "pretraining":
            logg.info("Model is in pretraining stage...")
            trainer_kwargs["early_stopping_monitor"] = 'elbo_validation'
        elif self.model_stage == "refining":
            logg.info("Model is in refining stage...")
            trainer_kwargs["early_stopping_monitor"] = "validation_loss"
        else:
            raise ValueError("Seems model is not in pretraining or refining")

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_state_assignment(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        hard_assignment: bool = False,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats

    @torch.inference_mode()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns the cells by genes latent time.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        velocity_mode = self.module.mode
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                if velocity_mode == "global":
                    prob_t = generative_outputs["prob_t"]
                    output = (prob_t * 20)
                    output = output.cpu().numpy()
                else:
                    pi = generative_outputs["px_pi"]
                    ind_prob = pi[..., 0]
                    rep_prob = pi[..., 1]
                    rep_steady_prob = pi[..., 2]
                    switch_time = F.softplus(self.module.switch_time_unconstr)

                    ind_time = generative_outputs["px_rho"] * switch_time
                    rep_time = switch_time + (
                        generative_outputs["px_tau"] * (self.module.t_max - switch_time)
                    )

                    if time_statistic == "mean":
                        output = (
                            ind_prob * ind_time
                            + rep_prob * rep_time
                            # + steady_prob * switch_time
                            # + rep_steady_prob * self.module.t_max
                        )
                    else:
                        t = torch.stack(
                            [
                                ind_time,
                                switch_time.expand(ind_time.shape),
                                rep_time,
                                torch.zeros_like(ind_time),
                            ],
                            dim=2,
                        )
                        max_prob = torch.amax(pi, dim=-1)
                        max_prob = torch.stack([max_prob] * 4, dim=2)
                        max_prob_mask = pi.ge(max_prob)
                        output = (t * max_prob_mask).sum(dim=-1)

                    output = output[..., gene_mask]
                    output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            if velocity_mode == "global":
                return pd.DataFrame(
                times,
                index=adata.obs_names[indices],
                )

            else:
                return pd.DataFrame(
                    times,
                    columns=adata.var_names[self.module.target_index],
                    index=adata.obs_names[indices],
                )
        else:
            return times

    @torch.inference_mode()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        velocity_mode = self.module.mode
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        velos = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                if velocity_mode == "global":
                    prob_t = generative_outputs["prob_t"]
                    beta = inference_outputs["beta"]
                    gamma = inference_outputs["gamma"]

                    t = prob_t * self.module.t_max
                    n_cells = len(prob_t)
                    mean_u, mean_s = self.module._get_global_unspliced_spliced(
                        t
                    )
                    if velo_mode == "spliced":
                        velo = beta * mean_u - gamma * mean_s
                    else:
                        NotImplementedError
                    output = velo
                else:
                    pi = generative_outputs["px_pi"]
                    beta = inference_outputs["beta"]
                    gamma = inference_outputs["gamma"]
                    tau = generative_outputs["px_tau"]
                    rho = generative_outputs["px_rho"]

                    ind_prob = pi[..., 0]
                    #steady_prob = pi[..., 1]
                    rep_prob = pi[..., 1]
                    switch_time = F.softplus(self.module.switch_time_unconstr)

                    ind_time = switch_time * rho
                    u_0, s_0 = self.module._get_induction_unspliced_spliced(
                        switch_time.reshape(1,len(switch_time))
                    )
                    rep_time = (self.module.t_max - switch_time) * tau
                    mean_u_rep, mean_s_rep = self.module._get_repression_unspliced_spliced(
                        u_0,
                        s_0,
                        beta,
                        gamma,
                        rep_time,
                    )
                    if velo_mode == "spliced":
                        velo_rep = beta * mean_u_rep - gamma * mean_s_rep
                    else:
                        velo_rep = -beta * mean_u_rep
                    mean_u_ind, mean_s_ind = self.module._get_induction_unspliced_spliced(
                        ind_time
                    )
                    if velo_mode == "spliced":
                        velo_ind = beta * mean_u_ind - gamma * mean_s_ind
                    else:
                        transcription_rate = alpha_1 - (alpha_1 - alpha) * torch.exp(
                            -lambda_alpha * ind_time
                        )
                        velo_ind = transcription_rate - beta * mean_u_ind

                    if velo_mode == "spliced":
                        # velo_steady = beta * u_0 - gamma * s_0
                        velo_steady = torch.zeros_like(velo_ind)
                    else:
                        # velo_steady = alpha - beta * u_0
                        velo_steady = torch.zeros_like(velo_ind)

                    # expectation
                    if velo_statistic == "mean":
                        output = (
                            ind_prob * velo_ind
                            + rep_prob * velo_rep
                        )
                    # maximum
                    else:
                        v = torch.stack(
                            [
                                velo_ind,
                                velo_steady.expand(velo_ind.shape),
                                velo_rep,
                                torch.zeros_like(velo_rep),
                            ],
                            dim=2,
                        )
                        max_prob = torch.amax(pi, dim=-1)
                        max_prob = torch.stack([max_prob] * 4, dim=2)
                        max_prob_mask = pi.ge(max_prob)
                        output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        spliced = spliced[:,:int(spliced.shape[1]/2)]
        spliced = spliced[:,self.module.target_index]

        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[:int(adata.shape[1]/2)][self.module.target_index],
                index=adata.obs_names[indices],
            )
        else:
            return velos

    @torch.inference_mode()
    def get_pred_readout(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        pre_u_all = []
        pre_s_all = []
        for tensors in scdl:
            minibatch_samples_u = []
            minibatch_samples_s = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                beta = inference_outputs["beta"]
                gamma = inference_outputs["gamma"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]

                ind_prob = pi[..., 0]
                #steady_prob = pi[..., 1]
                rep_prob = pi[..., 1]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = switch_time * rho
                u_0, s_0 = self.module._get_induction_unspliced_spliced(
                    switch_time.reshape(1,len(switch_time))
                )
                rep_time = (self.module.t_max - switch_time) * tau
                mean_u_rep, mean_s_rep = self.module._get_repression_unspliced_spliced(
                    u_0,
                    s_0,
                    beta,
                    gamma,
                    rep_time,
                )
                
                mean_u_ind, mean_s_ind = self.module._get_induction_unspliced_spliced(
                    ind_time
                )

                # expectation
                pre_u = (
                    ind_prob * mean_u_ind
                    + rep_prob * mean_u_rep
                )

                pre_s = (
                    ind_prob * mean_s_ind
                    + rep_prob * mean_s_rep
                )

                pre_u = pre_u[..., gene_mask]
                pre_u = pre_u.cpu().numpy()
                pre_s = pre_s[..., gene_mask]
                pre_s = pre_s.cpu().numpy()

                minibatch_samples_u.append(pre_u)
                minibatch_samples_s.append(pre_s)
            # samples by cells by genes
            pre_u_all.append(np.stack(minibatch_samples_u, axis=0))
            pre_s_all.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                pre_u_all[-1] = np.mean(pre_u_all[-1], axis=0)
                pre_s_all[-1] = np.mean(pre_s_all[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            pre_u_all = np.concatenate(pre_u_all, axis=-2)
            pre_s_all = np.concatenate(pre_s_all, axis=-2)
        else:
            pre_u_all = np.concatenate(pre_u_all, axis=0)
            pre_s_all = np.concatenate(pre_s_all, axis=0)

        if return_numpy is None or return_numpy is False:
            pre_u_all = pd.DataFrame(
                pre_u_all,
                columns=adata.var_names[self.module.target_index],
                index=adata.obs_names[indices],
            )
            pre_s_all = pd.DataFrame(
                pre_s_all,
                columns=adata.var_names[self.module.target_index],
                index=adata.obs_names[indices],
            )
            return pre_u_all, pre_s_all
        else:
            return pre_u_all, pre_s_all
            
    @torch.inference_mode()
    def get_ind_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns the cells by genes latent time.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                rep_prob = pi[..., 1]
                rep_steady_prob = pi[..., 2]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.module.t_max - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_time
                        # + steady_prob * switch_time
                        # + rep_steady_prob * self.module.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[self.module.target_index],
                index=adata.obs_names[indices],
            )
        else:
            return times

    @torch.inference_mode()
    def get_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        restrict_to_latent_dim: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s = []
        fits_u = []
        for tensors in scdl:
            minibatch_samples_s = []
            minibatch_samples_u = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                    generative_kwargs={"latent_dim": restrict_to_latent_dim},
                )

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s,
                    mixture_dist_u,
                    _,
                ) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                    alpha_1,
                    lambda_alpha,
                )
                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u

    @torch.inference_mode()
    def get_gene_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the likelihood per gene. Higher is better.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        rls = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                spliced = tensors[REGISTRY_KEYS.X_KEY]
                unspliced = tensors[REGISTRY_KEYS.U_KEY]

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s,
                    mixture_dist_u,
                    _,
                ) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                    alpha_1,
                    lambda_alpha,
                )
                device = gamma.device
                reconst_loss_s = -mixture_dist_s.log_prob(spliced.to(device))
                reconst_loss_u = -mixture_dist_u.log_prob(unspliced.to(device))
                output = -(reconst_loss_s + reconst_loss_u)
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls

    @torch.inference_mode()
    def get_rates(self):
        gamma, beta, alpha_1= self.module._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "alpha_1": alpha_1.cpu().numpy(),
        }

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        readout_layer: str,
        **kwargs,
    ) -> Optional[AnnData]:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        spliced_layer
            Layer in adata with spliced normalized expression
        unspliced_layer
            Layer in adata with unspliced normalized expression.

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, readout_layer, is_count_data=False)
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def get_directional_uncertainty(
        self,
        adata: Optional[AnnData] = None,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
    ):
        adata = self._validate_anndata(adata)

        logger.info("Sampling from model...")
        velocities_all = self.get_velocity(
            n_samples=n_samples, return_mean=False, gene_list=gene_list
        )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = _compute_directional_statistics_tensor(
            tensor=velocities_all, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        return df, cosine_sims

    def get_permutation_scores(
        self, labels_key: str, adata: Optional[AnnData] = None
    ) -> Tuple[pd.DataFrame, AnnData]:
        """Compute permutation scores.

        Parameters
        ----------
        labels_key
            Key in adata.obs encoding cell types
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.

        Returns
        -------
        Tuple of DataFrame and AnnData. DataFrame is genes by cell types with score per cell type.
        AnnData is the permutated version of the original AnnData.
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if labels_key not in adata.obs:
            raise ValueError(f"{labels_key} not found in adata.obs")

        # shuffle spliced then unspliced
        bdata = self._shuffle_layer_celltype(
            adata_manager, labels_key, REGISTRY_KEYS.X_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.U_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)

        ms_ = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_ = adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        ms_p = bdata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        mu_p = bdata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        spliced_, unspliced_ = self.get_expression_fit(adata, n_samples=10)
        root_squared_error = np.abs(spliced_ - ms_)
        root_squared_error += np.abs(unspliced_ - mu_)

        spliced_p, unspliced_p = self.get_expression_fit(bdata, n_samples=10)
        root_squared_error_p = np.abs(spliced_p - ms_p)
        root_squared_error_p += np.abs(unspliced_p - mu_p)

        celltypes = np.unique(adata.obs[labels_key])

        dynamical_df = pd.DataFrame(
            index=adata.var_names,
            columns=celltypes,
            data=np.zeros((adata.shape[1], len(celltypes))),
        )
        N = 200
        for ct in celltypes:
            for g in adata.var_names.tolist():
                x = root_squared_error_p[g][adata.obs[labels_key] == ct]
                y = root_squared_error[g][adata.obs[labels_key] == ct]
                ratio = ttest_ind(x[:N], y[:N])[0]
                dynamical_df.loc[g, ct] = ratio

        return dynamical_df, bdata

    def _shuffle_layer_celltype(
        self, adata_manager: AnnDataManager, labels_key: str, registry_key: str
    ) -> AnnData:
        """Shuffle cells within cell types for each gene."""
        from scvi.data._constants import _SCVI_UUID_KEY

        bdata = adata_manager.adata.copy()
        labels = bdata.obs[labels_key]
        del bdata.uns[_SCVI_UUID_KEY]
        self._validate_anndata(bdata)
        bdata_manager = self.get_anndata_manager(bdata)

        # get registry info to later set data back in bdata
        # in a way that doesn't require actual knowledge of location
        unspliced = bdata_manager.get_from_registry(registry_key)
        u_registry = bdata_manager.data_registry[registry_key]
        attr_name = u_registry.attr_name
        attr_key = u_registry.attr_key

        for lab in np.unique(labels):
            mask = np.asarray(labels == lab)
            unspliced_ct = unspliced[mask].copy()
            unspliced_ct = np.apply_along_axis(
                np.random.permutation, axis=0, arr=unspliced_ct
            )
            unspliced[mask] = unspliced_ct
        # e.g., if using adata.X
        if attr_key is None:
            setattr(bdata, attr_name, unspliced)
        # e.g., if using a layer
        elif attr_key is not None:
            attribute = getattr(bdata, attr_name)
            attribute[attr_key] = unspliced
            setattr(bdata, attr_name, attribute)

        return bdata


def _compute_directional_statistics_tensor(
    tensor: np.ndarray, n_jobs: int, n_cells: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n_cells))
    df["directional_variance"] = np.nan
    df["directional_difference"] = np.nan
    df["directional_cosine_sim_variance"] = np.nan
    df["directional_cosine_sim_difference"] = np.nan
    df["directional_cosine_sim_mean"] = np.nan
    logger.info("Computing the uncertainties...")
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_directional_statistics_per_cell)(tensor[:, cell_index, :])
        for cell_index in range(n_cells)
    )
    # cells by samples
    cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
    df.loc[:, "directional_cosine_sim_variance"] = [
        results[i][1] for i in range(n_cells)
    ]
    df.loc[:, "directional_cosine_sim_difference"] = [
        results[i][2] for i in range(n_cells)
    ]
    df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
    df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
    df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

    return df, cosine_sims


def _directional_statistics_per_cell(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal function for parallelization.

    Parameters
    ----------
    tensor
        Shape of samples by genes for a given cell.
    """
    n_samples = tensor.shape[0]
    # over samples axis
    mean_velocity_of_cell = tensor.mean(0)
    cosine_sims = [
        _cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
    ]
    angle_samples = [np.arccos(el) for el in cosine_sims]
    return (
        cosine_sims,
        np.var(cosine_sims),
        np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
        np.var(angle_samples),
        np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
        np.mean(cosine_sims),
    )


def _centered_unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the centered unit vector of the vector."""
    vector = vector - np.mean(vector)
    return vector / np.linalg.norm(vector)


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns cosine similarity of the vectors."""
    v1_u = _centered_unit_vector(v1)
    v2_u = _centered_unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
