import os
from typing import Optional

import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_VAE_MLP
from .denoising_ae_config import denoising_AEConfig
from torch import tensor, cat, exp, std
import torch
from numpy.random import binomial
import numpy as np

class denoising_AE(BaseAE):
    """

    Args:
        model_config (AEConfig): The Autoencoder configuration setting the main parameters of the
            model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: denoising_AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "denoising_AE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_VAE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        self.p = 0.1 #p for the bernoulli encoding the number of missing values
        self.nrepeats = 10
        self.init_loss = True
        self.beta = 1

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
            - nU is the number of time we wish to makes copies of x and apply a different U
            When self.p>0, we repeat nU times x and apply a different u everytime

        """

        n_repeats = self.n_repeats if self.training else 3

        x = inputs["data"]
        z = self.encoder(x).embedding

        ## NORMAL vAE step

        #if self.p>0: #augment data
        X, Z = x.repeat_interleave(n_repeats,dim=0), z.repeat_interleave(n_repeats,dim=0)
        XV = self.corrupt(X) #corrupt by adding missing values
        ZV = self.encoder(XV) #encoding of xU
        ZV_mu, ZV_noise = ZV.embedding, ZV.log_covariance

        XV_hat = self.decoder(ZV_mu)["reconstruction"]
        if XV_hat.shape != X.shape:
            XV_hat = torch.squeeze(XV_hat, 1) #if dimension added
        recon_loss = self.loss_function(XV_hat[(X!=-10)], X[(X!=-10)])

        variance_loss = torch.std(ZV_mu,axis=0)

        #hsic_loss = self.HSIC(ZV_mu,ZV_noise)

        loss = recon_loss + self.beta*variance_loss #+ self.gamma*hsic_loss

        output = ModelOutput(loss=loss, recon_x=XV_hat, z=Z)


        return output


    def add_missing_values(self, x, u, nU):
        """
        expands x nU times, adds missing values folllowing a new mask U for each of the nU repeats
        """
        x_nU, u_nU = x.repeat_interleave(nU,dim=0), u.repeat_interleave(nU,dim=0)
        #set xU as nU repeat of x with a different U applied each time
        xU = x_nU.detach().clone()
        binomial_probas = u_nU.detach().cpu().numpy()*self.p #we want to delete each feat with proba self.ps
        U = tensor(binomial(n=1,p=binomial_probas,size=xU.shape))
        xU[U==1] = -10

        return xU, x_nU, u_nU

    def corrupt(self,data):

        noise = torch.bernoulli(torch.ones(data.shape)*self.p) * torch.rand(data.shape)*torch.max(data)

        return data + noise

    def loss_function(self, recon_x, x):

        MSE = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)
        return MSE.mean(dim=0)

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = AEConfig.from_json_file(path_to_model_config)

        return model_config
