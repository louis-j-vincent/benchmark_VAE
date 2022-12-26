import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset
from .. import AE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from .emae_config import EMAE_Config
from kmeans_pytorch import kmeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F

#from sklearn.cluster import KMeans
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
        self.beta = 1
        self.temperature = 0
        self.Z, self.labels = None, None
        self.variationnal = True
        self.K = 10 #nb of Gaussians
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.mu = torch.rand((self.K,model_config.latent_dim)).to(device)
        self.mu = torch.zeros((self.K,model_config.latent_dim)).to(device)
        for k in range(self.K):
            self.mu[k,k] = 1.
        self.init = True
        self.Sigma = torch.ones(self.mu.shape).to(device)
        self.quantile = 1
        #self.Sigma = torch.eye(model_config.latent_dim).repeat(self.K,1,1).to(device)
        self.device = device
        self.alpha = (torch.ones(self.K)/self.K).to(device) #p probabilities for each gaussian 
        self.hist = {'mu':self.mu, 'Sigma':self.Sigma}
        self.plot = False

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded

        Args:
            inputs (BaseDataset): An instance of pythae's datasets

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        y = F.one_hot(inputs["labels"].to(torch.int64),num_classes=self.K).float().to(self.device)

        z = self.encoder(x).embedding
        if self.variationnal:
            sigma_small = y@self.Sigma*torch.abs((self.quantile + y@self.mu)/(1.96 + z))
            sigma_small = torch.maximum(torch.ones(sigma_small.shape)*0.01, sigma_small)
            z += torch.normal(torch.zeros(z.shape).to(self.device),sigma_small).to(self.device)

        if self.Z is None:
            self.Z = z
            self.labels = y
        else:
            self.Z = torch.cat((self.Z, z),0)
            self.labels = torch.cat((self.labels, y),0)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, embedding_loss = self.loss_function(recon_x, x, z)

        #self.update_parameters(z)
    
        #self.temperature = min(0.2,self.temperature)
        #self.temperature = 0.01
        if self.beta>0 and self.temperature > 0.1:
            LLloss, sep_loss = self.likelihood_loss(z,y)
            sep_loss = 0
            loss = recon_loss + (sep_loss + LLloss)*self.beta
            print(recon_loss, embedding_loss, LLloss,loss)
        else:
            loss = recon_loss
        #min_max_loss = self.min_max_loss(z,y)
        #loss += min_max_loss

        
        #print(recon_loss,embedding_loss,min_max_loss)

        output = ModelOutput(
            loss=loss,
            recon_loss=recon_loss,
            embedding_loss=embedding_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def likelihood_loss(self, Z, y):
        #E-step
        Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
        Sigma = self.Sigma[None,:,:] 
        N_log_prob = torch.minimum(-0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma)),torch.tensor(0) )#.sum(axis=0)
        N_prob = (torch.exp(N_log_prob) * y[:,:,None]).sum(axis=1) #only use the kth gaussian
        #log_prob = torch.log(prob).sum(axis=0) #sum on all patients
            
        prob = N_prob.mean()
        separation_prob = N_prob.prod() #prod on K gaussians

        #plt.hist((N_prob).detach().numpy().flatten())
        #plt.show()

        return 1 - prob.mean(), separation_prob

    def min_max_loss(self, Z, labels):
        """
        Loss to minimize variance within each cluster, maximize variance between the center of each cluster
        """
        mu_hat = (Z[:,None,:]*labels[:,:,None]).sum(axis=0) / labels[:,:,None].sum(axis=0)
        var_hat = ((( (Z- mu_hat[None,:,:])**2 )[:,None,:]*labels[:,:,None]).sum(axis=0) / labels[:,:,None].sum(axis=0)).mean()
        var_mu_hat = torch.var(mu_hat)

        return var_hat - var_mu_hat

    def detach_parameters(self):
        self.Z = self.Z.detach().clone()
        self.mu = self.mu.detach()
        self.Sigma = self.Sigma.detach()
        self.alpha = self.alpha.detach()

    def draw_95_ellipse(self, mu, Sigma, c="black", alpha=1, ax=None):
        if len(Sigma.shape) == 1:
            Sigma = torch.diag(Sigme)
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        axa = np.sqrt(eigenvalues[0] * 5.991)
        axb = np.sqrt(eigenvalues[1] * 5.991)
        va = eigenvectors[:, 0]
        vb = eigenvectors[:, 1]
        x = np.linspace(0, 2*np.pi, 100)
        trace = np.array([np.cos(e)*va*axa + np.sin(e)*vb*axb + mu for e in x])
        ax.plot(trace[:,0], trace[:,1], c=c, alpha=alpha, linewidth=3)

        return ax 

    def update_parameters(self,Z=None):
        if Z is None:
            Z = self.Z
        if self.init==True:
            self.mu = torch.matmul(self.labels.T,Z)
            self.alpha = self.labels.mean(axis=0)
            self.init=False
        print('Updating parameters')
        for i in range(1):

            #E-step
            tau = self.labels
            print(tau.mean())

            # M-step
            tau_sum = tau[:,:,None].sum(axis=0).detach()
            self.mu = (tau[:,:,None]*Z[:,None,:]).sum(axis=0).detach()/tau_sum
            self.Sigma = (tau[:,:,None] * (Z[:,None,:]-self.mu[None,:,:])**2).sum(axis=0).detach()/tau_sum

        if self.plot==True:
            X = Z.detach().numpy()
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            fig, ax = plt.subplots(1,1,figisze=(18,8))
            ax.scatter(X[:,0],X[:,1],c=torch.where(self.labels==1)[1].detach())
            for i in range(self.K):
                mu = (self.mu[i]@pca.components_.T).detach().numpy()
                Sigma = (pca.components_@torch.diag(self.Sigma[i]).detach().numpy()@pca.components_.T)
                ax = self.draw_95_ellipse(mu, Sigma, alpha = float(self.alpha[i].detach()), ax=ax)
            fig.save(f'plot{epoch}.png')
        self.Z = None
        self.hist['mu'] = torch.vstack([self.hist['mu'],self.mu])
        self.hist['Sigma'] = torch.vstack([self.hist['Sigma'],self.Sigma])

    ## OLD

    def likelihood_loss_multidim(self, X):
        Y = (X[:,None,:]-self.mu[None,:,:])
        self.Lambda = torch.linalg.inv(self.Sigma)
        log = torch.einsum("ikp, kpq, ikq -> ik", Y, self.Lambda, Y) 
        d = self.mu.shape[1]
        N_prob = torch.exp(-log/2) * torch.sqrt(torch.det(self.Lambda)) / (2*torch.pi)**(d/2)
        prob = (N_prob * self.alpha).sum(axis=1)
        prob = torch.maximum(torch.minimum(prob,torch.tensor(1)),torch.tensor(0))

        separation_prob = N_prob.prod(axis=1).mean()

        return 1 - prob.mean(), separation_prob

    def log_likelihood(self, X, mu, Sigma, alpha):
        Y = (X[:,None,:]-self.mu[None,:,:])
        self.Lambda = torch.linalg.inv(Sigma)
        print(self.Lambda.shape)
        log = torch.einsum("ikp, kpq, ikq -> ik", Y, self.Lambda, Y) 
        N_prob = torch.exp(-log/2) / (2*torch.pi*torch.sqrt(torch.det(Sigma)))
        print(prob)
        return torch.fmax(torch.log(prob).sum(),torch.tensor(-10**3))

    def ll(self, X):
        #E-step
        Y = (X[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
        Sigma = self.Sigma[None,:,:] 
        N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma) )#.sum(axis=0)
        tau = torch.exp(N_log_prob.sum(axis=2) * self.alpha[None,:]) #sum_k p(x_i ; z_i = k) p(z_i = k) = p(x_i)

    def update_parameters_noprior(self,Z=None,final_update=False):
        if Z is None:
            Z = self.Z
            final_update = True
        if self.init==True:
            cids, ccenters = kmeans(X=Z, num_clusters=self.K, device=self.device)
            self.mu = ccenters
            self.init=False
        print('Updating parameters')
        for i in range(1):

            #E-step
            Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
            Sigma = self.Sigma[None,:,:] 
            print(Y.shape, Sigma.shape, Z.shape, self.mu.shape)
            N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma) )#.sum(axis=0)
            log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2)
            log_tau = log_tau - torch.logsumexp(log_tau, axis=1)[:,None]
            tau = torch.exp(log_tau) #p(x_i ; z_i = k) p(z_i = k)
            print(tau.mean(),'tau mean')
            #yprint(tau.sum(axis=1),tau.mean(axis=0))

            # M-step
            self.mu = torch.exp(torch.logsumexp(torch.log(Z[:,None,:]) + log_tau[:,:,None],axis=0) - torch.logsumexp(log_tau[:,:,None],axis=0)).detach()
            #self.mu = (Z[:,None,:] * tau[:,:,None]).mean(axis=0).detach()

            print(Y.shape, tau.shape)
            self.Sigma = ( Y**2 * tau[:,:,None] ).mean(axis=0).detach()
            self.alpha = tau.mean(axis=0).detach()
            self.alpha /= self.alpha.sum() # Regularize result

            #_=self.log_likelihood(self.Z,self.mu,self.Sigma,self.alpha)
        X = Z.detach().numpy()
        pca = PCA(n_components=2)
        pca.fit(self.mu)
        X = pca.transform(X)
        plt.scatter(X[:,0],X[:,1])
        for i in range(self.K):
            mu = (self.mu[i]@pca.components_.T).detach().numpy()
            Sigma = (pca.components_@torch.diag(self.Sigma[i]).detach().numpy()@pca.components_.T)
            self.draw_95_ellipse(mu, Sigma, alpha = float(self.alpha[i].detach()))
        plt.show()
        if final_update:
            self.Z = None

    def update_parameters_multidim(self,Z=None):
        if Z is None:
            Z = self.Z
        if self.init==True:
            cids, ccenters = kmeans(X=Z, num_clusters=self.K, device=self.device)
            #print(cids.shape,ccenters.shape)
            self.mu = ccenters
            self.init=False
        print('Updating parameters')
        for i in range(10):
            self.Lambda = torch.linalg.inv(self.Sigma)
            Y = (Z[:,None,:]-self.mu[None,:,:]).detach()
            log = torch.einsum("ikp, kpq, ikq -> ik", Y, self.Lambda, Y).detach()
            logdet = torch.logdet(self.Sigma).detach()
            #_, logdet = np.linalg.slogdet(Sigma)
            N_log_prob = -log/2 - torch.log(2*torch.tensor(torch.pi))/2 - logdet/2
            print(N_log_prob.shape)
            log_tau = torch.log(self.alpha+1e-5)+N_log_prob
            log_tau = log_tau - torch.logsumexp(log_tau, axis=1)[:,None]
            tau = torch.exp(log_tau)

            # M-step
            self.mu = (torch.einsum("ik, ip -> kp", tau, Z) / (tau.sum(axis=0)[:,None] + 1e-5)).detach()
            Y = (Z[:,None,:]-self.mu[None,:,:])
            self.Sigma = (1e-4 * torch.eye(Z.shape[-1])[None,:,:].to(self.device) + torch.einsum("ikp, ikq, ik -> kpq", Y, Y, tau) / (tau.sum(axis=0)[:,None,None] + 1e-5)).detach()
            self.alpha = tau.mean(axis=0).detach()
            self.alpha /= self.alpha.sum() # Regularize result

            #_=self.log_likelihood(self.Z,self.mu,self.Sigma,self.alpha)
        X = Z.detach().numpy()
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        plt.scatter(X[:,0],X[:,1])
        for i in range(self.K):
            mu = (self.mu[i]@pca.components_.T).detach().numpy()
            Sigma = (pca.components_@self.Sigma[i].detach().numpy()@pca.components_.T)
            fig = self.draw_95_ellipse(mu, Sigma, alpha = float(self.alpha[i].detach()))
        
        plt.show()
        #fg.show()
        #fg.save('fig.png')
        self.Z = None

    def loss_function(self, recon_x, x, z):

        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).mean(dim=-1) #was sum before I modified

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
