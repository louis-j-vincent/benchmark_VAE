import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from .. import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .emae_config import EMAE_Config


class EMAE(AE):
    """Regularized Autoencoder with L2 decoder params regularization model.

    Args:
        model_config (RAE_L2_Config): The Autoencoder configuration setting the main parameters of the
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
        model_config: EMAE_Config,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        AE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "EMAE"
        self.Zs = None

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]

        z = self.encoder(x).embedding
        if torch.Zs is None:
            torch.Zs = z
        else:
            self.Zs = torch.cat((torch.Zs, z),0)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, embedding_loss = self.loss_function(recon_x, x, z)

        log_likelihood_loss = self.log_likelihood(z,self.mu,self.Sigma,self.alpha)

        loss += log_likelihood_loss

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            embedding_loss=embedding_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def log_likelihood(self, X, mu, Sigma, alpha):
        Y = (X[:,None,:]-mu[None,:,:])
        self.Lambda = torch.inverse(Sigma)
        log = torch.einsum("ikp, kpq, ikq -> ik", Y, self.Lambda, Y)
        N_prob = torch.exp(-log/2) / (2*torch.pi*torch.sqrt(torch.det(Sigma)))
        prob = (N_prob * alpha).sum(axis=1)
        return torch.log(prob).sum()

    def update_parameters(self):
        Y = (self.Z[:,None,:]-mu[None,:,:])
        log = torch.einsum("ikp, kpq, ikq -> ik", Y, self.Lambda, Y)
        logdet = torch.logdet(self.Sigma)
        #_, logdet = np.linalg.slogdet(Sigma)
        N_log_prob = -log/2 - torch.log(2*torch.pi)/2 - logdet/2
        log_tau = torch.log(alpha+1e-5)+N_log_prob
        log_tau = log_tau - torch.logsumexp(log_tau, axis=1)[:,None]
        tau = torch.exp(log_tau)
        

        # M-step
        self.mu = torch.einsum("ik, ip -> kp", tau, self.Z) / (tau.sum(axis=0)[:,None] + 1e-5)
        Y = (self.Z[:,None,:]-mu[None,:,:])
        self.Sigma = 1e-4 * torch.eye(d)[None,:,:] + torch.einsum("ikp, ikq, ik -> kpq", Y, Y, tau) / (tau.sum(axis=0)[:,None,None] + 1e-5)
        self.alpha = tau.mean(axis=0)
        self.alpha /= alpha.sum() # Regularize result
        self.Z = None

    def loss_function(self, recon_x, x, z):

        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2

        return (
            (recon_loss + self.model_config.embedding_weight * embedding_loss).mean(
                dim=0
            ),
            (recon_loss).mean(dim=0),
            (embedding_loss).mean(dim=0),
        )

    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = RAE_L2_Config.from_json_file(path_to_model_config)

        return model_config
