import os
from typing import Optional

import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_AE_MLP
from .ae_config import AEConfig
from torch import tensor, cat, exp, std
import torch
from numpy.random import binomial


class AE_multi_U(BaseAE):
    """Vanilla Autoencoder model.

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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE_multi_U"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        self.p = 0 #p for the bernoulli encoding the number of missing values
        self.nU = 10
        self.is_merging_net = False

    def add_missing_values(self, x, u, nU):
        """
        expands x nU times, adds missing values folllowing a new mask U for each of the nU repeats
        """
        x_nU, u_nU = x.repeat((nU,1)), u.repeat((nU,1))
        #set xU as nU repeat of x with a different U applied each time
        xU = x_nU.detach().clone()
        binomial_probas = (u_nU)*self.p #we want to delete each feat with proba self.ps
        U = tensor(binomial(n=1,p=binomial_probas,size=xU.shape))
        xU[U==1] = -10

        return xU, x_nU, u_nU

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
            - nU is the number of time we wish to makes copies of x and apply a different U
            When self.p>0, we repeat nU times x and apply a different u everytime

        """

        x = inputs["data"]
        u = (x!=-10)
        nU = self.nU if self.training else 3
        z = self.encoder(x).embedding 

        if self.p>0:
            xU, x_repeat, u_repeat = self.add_missing_values(x, u, nU)
            zU = self.encoder(xU).embedding #encoding of xU

        #if SNDS-like data, get z_alpha the equivalent of zU for SNDS-like data
        is_x_alpha = (inputs['labels']==1.).float().mean()!=1.
        if is_x_alpha:
            x_alpha = inputs["labels"]
            z_alpha = self.encoder.encoder_alpha(x_alpha)#.embedding
            z_alpha = z_alpha.repeat((nU,1))

        #Concatenate zU and z_alpha to get information from both for Z
        if self.is_merging_net:
            z = cat((zU,z_alpha),axis=1)
            z = self.encoder.merging_net(z)
        else:
            z = zU

        recon_x = self.decoder(z)["reconstruction"]
        loss = self.loss_function(recon_x[u_repeat], x_repeat[u_repeat])
        
        if is_x_alpha:
            loss += self.loss_function(z_alpha,zU) 
        else:
            loss += self.loss_function(z_repeat,zU) # + exp(-self.loss_function(z,z*0.))

        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

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

class vAE(BaseAE): #equivalent of AE_multi_U_w_variance
    """Vanilla Autoencoder model.

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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE_multi_U"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        self.p = 0.1 #p for the bernoulli encoding the number of missing values
        self.nU = 10
        self.init_loss = True

    def add_missing_values(self, x, u, nU):
        """
        expands x nU times, adds missing values folllowing a new mask U for each of the nU repeats
        """
        x_nU, u_nU = x.repeat((nU,1)), u.repeat((nU,1))
        #set xU as nU repeat of x with a different U applied each time
        xU = x_nU.detach().clone()
        binomial_probas = (u_nU)*self.p #we want to delete each feat with proba self.ps
        U = tensor(binomial(n=1,p=binomial_probas,size=xU.shape))
        xU[U==1] = -10

        return xU, x_nU, u_nU

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
            - nU is the number of time we wish to makes copies of x and apply a different U
            When self.p>0, we repeat nU times x and apply a different u everytime

        """

        x = inputs["data"]
        u = (x!=-10)
        nU = self.nU if self.training else 3
        z = self.encoder(x).embedding 

        if self.p>0:
            xU, x_repeat, u_repeat = self.add_missing_values(x, u, nU)
            z_out = self.encoder(xU) #encoding of xU
            zU_mu, zU_sigma = z_out.embedding, z_out.log_covariance

        z_anchor = z.repeat((nU,1))
        recon_x = self.decoder(zU_mu)["reconstruction"]

        recon_loss = self.loss_function(recon_x[u_repeat], x_repeat[u_repeat])
        if self.init_loss:
            reg_loss = self.loss_function(z_anchor, zU_mu)
        else:
            reg_loss = self.loss_log_proba(z_anchor, zU_mu, zU_sigma)

        loss = recon_loss + 0.1*reg_loss

        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

    def loss_log_proba(self, x, mu, sigma):
        sigma = torch.abs(sigma)
        eps = torch.tensor(1e-5)
        loss_mu = ( (x-mu)**2 / (sigma + eps) ).sum(axis=1).mean(axis=0)
        loss_sigma = torch.maximum(torch.tensor(0), torch.log(sogma.mean(axis=1) + eps)).mean(axis=0)
        loss = loss_mu + loss_sigma
        return loss

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

class AE(BaseAE):
    """Vanilla Autoencoder model.

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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        z = self.encoder(x).embedding
        recon_x = self.decoder(z)["reconstruction"]
        loss = self.loss_function(recon_x,x)

        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

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

class AE_Z_alpha(BaseAE):
    """Vanilla Autoencoder model.

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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE_Z_alpha"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        self.p = 0 #p for the bernoulli encoding the number of missing values

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        u = (x!=-10)
        xU = x.detach().clone()

        if self.p>0:
            # set some values to nan
            U = tensor(binomial(n=1,p=self.p,size=x.shape))
            xU[U==1] = -10
        
        z = self.encoder(xU).embedding
        recon_x = self.decoder(z)["reconstruction"]
        loss = self.loss_function(recon_x[u], x[u])

        if (inputs['labels']==1.).float().mean()!=1.:
            x_alpha = inputs["labels"]
            z_alpha = self.encoder.encoder_alpha(x_alpha)#.embedding
            loss += 0.1*self.loss_function(z,z_alpha) + exp(-self.loss_function(z_alpha,z_alpha*0.))
        
        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

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

class AE_Z_alpha2(BaseAE):
    """Vanilla Autoencoder model.

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
        model_config: AEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, decoder=decoder)

        self.model_name = "AE_Z_alpha2"

        if encoder is None:
            if model_config.input_dim is None:
                raise AttributeError(
                    "No input dimension provided !"
                    "'input_dim' parameter of BaseAEConfig instance must be set to 'data_shape' where "
                    "the shape of the data is (C, H, W ..). Unable to build encoder "
                    "automatically"
                )

            encoder = Encoder_AE_MLP(model_config)
            self.model_config.uses_default_encoder = True

        else:
            self.model_config.uses_default_encoder = False

        self.set_encoder(encoder)
        self.p = 0 #p for the bernoulli encoding the number of missing values
        self.is_merging_net=False

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        u = (x!=-10)
        xU = x.detach().clone()

        if self.p>0:
            # set some values to nan
            U = tensor(binomial(n=1,p=self.p,size=x.shape))
            xU[U==1] = -10
        
        z_eps = self.encoder(xU).embedding

        #if SNDS-like data
        is_x_alpha = (inputs['labels']==1.).float().mean()!=1.
        if is_x_alpha:
            x_alpha = inputs["labels"]
            z_alpha = self.encoder.encoder_alpha(x_alpha)#.embedding

        #use merging net
        if self.is_merging_net:
            z = cat((z_eps,z_alpha),axis=1)
            z = self.encoder.merging_net(z)
        else:
            z = z_eps

        recon_x = self.decoder(z)["reconstruction"]
        loss = self.loss_function(recon_x[u], x[u]) 
        
        if is_x_alpha:
            loss += self.loss_function(z_alpha,z_eps) #exp(-self.loss_function(z_alpha,z_alpha*0.)) + 0.1*
            #loss = self.loss_function(z_alpha,z_eps)

        output = ModelOutput(loss=loss, recon_x=recon_x, z=z)

        return output

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
