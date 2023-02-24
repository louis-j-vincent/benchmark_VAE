import os
from typing import Optional

import torch.nn.functional as F

from ...data.datasets import BaseDataset
from ..base import BaseAE
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..nn.default_architectures import Encoder_AE_MLP
from .vfemae_config import vfEMAEConfig
from torch import tensor, cat, exp, std
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d as GF1D


class vfEMAE(BaseAE): #equivalent of AE_multi_U_w_variance
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
        model_config: vfEMAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        BaseAE.__init__(self, model_config=model_config, encoder = encoder, decoder=decoder)

        self.model_name = "vfEMAE"

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
        ## PARAMS RELATIVE TO vAE
        self.p = 0.1 #p for the bernoulli encoding the number of missing values
        self.n_repeats = 10 #number of repetition
        self.init_loss = True
        ## PARAMS RELATIVE TO EMAE
        self.beta = 1
        self.gamma = 1
        self.EM_steps = 10
        self.temperature = 0.01
        self.variationnal = True
        self.quantile = torch.tensor(0.5)
        self.plot = False
        self.temp_start = -1
        self.use_missing_labels = False
        self.tempered_EM = False
        self.deterministic_EM = True
        self.print = False

        ##init EMAE params
        self.init = True
        self.Z, self.labels = None, None
        self.K = model_config.K #nb of Gaussians
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mu = torch.zeros((self.K,model_config.latent_dim)).to(device)
        print('self.K is ',self.K,self.mu.shape[1])
        print(min(self.K,self.mu.shape[1]))
        for k in range(min(self.K,self.mu.shape[1])):
            print(k)
            self.mu[k,k] = 1.
        self.Sigma = torch.ones(self.mu.shape).to(device)
        self.cluster_map = torch.eye(self.K)
        self.device = device
        self.alpha = (torch.ones(self.K)/self.K).to(device) #prior probabilities for each gaussian 
        self.hist = {'mu':self.mu, 'Sigma':self.Sigma, 'beta':[self.beta], 'tau':[]}
        self.recon_loss, self.ll_loss = None, None
        self.epoch = 0
        self.i_i50 = 18

        ##Rieman
        self.M = []
        self.centroids = []
        self.S = 1

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

        x, tempo = inputs["data"][:,1:], inputs['data'][:,0]
        z = self.encoder(x).embedding
        y_missing = F.one_hot(inputs["labels"].to(torch.int64),num_classes=self.K*3).float().to(self.device) #3 to overshoot if there are many classes
        y = y_missing[:,:self.K]

        if self.variationnal: #do a VAE-like step by sampling on a gaussian centered on z and "contained" inside the class gaussian
            ratio = ((self.quantile)/(1.96 + torch.abs(z - y@self.mu)))**2
            sigma_small_max = torch.fmax((y@self.Sigma**0.5)*ratio,torch.tensor(0.001)) 
            z_var = z + torch.normal(torch.zeros(z.shape).to(self.device),sigma_small_max).to(self.device)
        else:
            z_var = z

        ## NORMAL vAE step

        #if self.p>0: #augment data
        X, Z = x.repeat_interleave(n_repeats,dim=0), z.repeat_interleave(n_repeats,dim=0)
        XV = self.corrupt(X) #corrupt by adding missing values
        ZV = self.encoder(XV) #encoding of xU
        try:
            ZV_mu, ZV_sigma = ZV.embedding, torch.exp(ZV.log_covariance)
        except AttributeError:
            ZV_mu = ZV.embedding

        XV_hat = self.decoder(ZV_mu)["reconstruction"]
        if XV_hat.shape != X.shape:
            XV_hat = torch.squeeze(XV_hat, 1) #if dimension added
        recon_loss = self.loss_function(XV_hat[(X!=-10)], X[(X!=-10)])

        ## EMAE step
        #print(z[y_missing[:,0]==1].shape)
        #if (y_missing[:,0]==1).float().mean()!=0 and (y_missing[:,0]==1).float().mean()!=1:
        #    recon_loss += self.gamma/(self.loss_function(z[y_missing[:,0]==1].mean(axis=0).reshape(1,-1), z[y_missing[:,0]==0].mean(axis=0).reshape(1,-1)))


        if self.Z is None: #this will be true at the beginning of each epoch
            self.Z = z
            self.labels = y_missing
            self.tempo = tempo
        else:
            self.Z = torch.cat((self.Z, z),0)
            self.labels = torch.cat((self.labels, y_missing),0)
            self.tempo = torch.cat((self.tempo, tempo),0)

        delta_Z = torch.abs(ZV_mu[1:] - ZV_mu[:-1])
        delta_X = torch.abs(X[1:] - X[:-1])**2 + 0.0001
        U = (X==-10)
        delta_U = (1 - U[1:]*U[:-1]*1) + 0.0001
        temporal_loss = torch.abs( delta_Z.mean(axis=-1) * delta_U.sum(axis=-1) / (delta_X * delta_U).sum(axis=-1) ).mean() * (delta_X * delta_U).mean()
        # we divide (aka multiply here) by delta_U sum to get mean of delta_x * delta_U only where delta_U==1
        #print(temporal_loss)

        #if self.beta>0 and self.temperature > self.temp_start: 

        LLloss, classif_loss = self.likelihood_loss(z,y_missing) #Log likelihood loss on all data 
        #LLloss_true = self.likelihood_loss(z,y_missing,ll_with_missing_labels=False) #Log likelihood only on labelised data

        #LLloss_tempered = (LLloss*self.temperature  + (1-self.temperature)*LLloss_true)*self.temperature
        LLloss_tempered = LLloss + classif_loss

        if self.temperature > self.temp_start:
            loss = recon_loss + LLloss_tempered*self.beta*self.temperature
        else:
            loss = recon_loss
        #print to observe evolution of different losses
        loss += temporal_loss * self.gamma
        if self.print:
           print(f'Losses: e{self.epoch:.4f} - recon {recon_loss.item():.4f} - LLoss_total {LLloss.item():.4f} classif_loss {classif_loss.item():.4f} total_loss {loss.item():.4f}')
        output = ModelOutput(loss=recon_loss + self.beta*(LLloss + classif_loss) + temporal_loss*self.gamma, recon_x=XV_hat, z=Z)

        return output

    def corrupt(self, X):
        """
        Adds missing values with probability p
        """
        #if variance, then U should be divided then doubled up

        U = (X!=-10)*1 #get mask on missing values
        U_half = U[:,:U.shape[1]//2]
        V = torch.bernoulli( U_half*self.p )
        V = V.repeat(1,2)
        XV = X.detach().clone()
        XV[V==1] = -10 #add missing value when V mask == 1

        return XV

    def loss_function(self, recon_x, x):

        MSE = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)
        return MSE.mean(dim=0)

## compute likelihood loss

    def likelihood_loss(self, Z, y, ll_with_missing_labels = True):
        """
        Compute likelihood of gaussian Mixture Model by:
        - computing tau (if we are using missing labels) which is the a posteriori probability for each point of being a part of each of hte K clusters
        - computing the likelihood loss defined as sum_i sum_k tau_i,k * P(z_i sachant mu_k, Sigma_k)
        """

        tau = torch.clone(y[:,:self.K])#.detach().cpu()
        missing_labels = (y[:,self.K:].sum(axis=1)>0)

        #E-step
        Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
        Sigma = self.Sigma[None,:,:] + 0.0001
        N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma) )
        log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
        log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None])#.detach().cpu()

        loss_classif = F.l1_loss( tau[~missing_labels], torch.exp(log_tau[~missing_labels]) ) * self.K

        tau = torch.exp(log_tau)

        #get log prob
        N_log_prob = torch.minimum(-0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma)),torch.tensor(0) )#.sum(axis=0)
        N_prob = (torch.exp(N_log_prob) * tau[:,:self.K,None]).sum(axis=1) #only use the kth gaussian 
        prob = torch.mean(N_prob[N_prob==N_prob]) #nanmean walkaround

        pk_prior = ((tau[missing_labels].mean(axis=0) - y[:,:self.K].mean(axis=0))**2 / 2 ).sum()

        prob = prob - pk_prior

        return 1 - prob.mean(), loss_classif

    def tau_loss(self,z,y):
        tau = self.compute_tau(z,y,replace_y=True)
        loss = F.l1_loss(tau,y)
        return loss

    def compute_tau(self, Z, y, replace_y = False):
        """
        Compute probability for each point to be a aprt of each gaussian
        """

        tau = torch.clone(y[:,:self.K])#.detach().cpu()
        missing_labels = torch.where(y[:,self.K:].sum(axis=1)>0)[0]#.detach().cpu()

        #E-step
        Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
        Sigma = self.Sigma[None,:,:] 
        N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma) )#.detach().cpu()
        log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
        log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None])#.detach().cpu()
        if replace_y:
            tau = torch.exp(log_tau)#
        else:
            tau[missing_labels] = torch.exp(log_tau[missing_labels])#

        return tau

## update parameters

    def update_parameters(self,Z=None):
        """
        Run self.EM_steps (set to 1) steps of the EM algorithm to estimate parameters 
        """
        labels = self.labels[:,:self.K]

        if Z is None:
            Z = self.Z
        if self.init==True:
            self.mu = torch.matmul(labels.T,Z)
            self.alpha = torch.mean(labels[labels==labels],axis=0) + (1/self.K)*(self.labels[:,self.K:].sum(axis=1).mean())
            self.init=False

        print('Updating parameters')
        for i in range(self.EM_steps):
            #E-step
            tau = torch.clone(labels).detach().cpu()
            #infer missing labels
            missing_labels = torch.where(self.labels[:,self.K:].sum(axis=1)>0)[0].detach().cpu()
            Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
            Sigma = self.Sigma[None,:,:]  + 0.0001
            N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma) )#.detach().cpu()
            log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
            log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None]).detach().cpu()
            tau[missing_labels] = torch.exp(log_tau[missing_labels]) 
            #valid_keys = (log_tau[missing_labels] == log_tau[missing_labels]) #don't replace nan values
            #tau[missing_labels][valid_keys] = torch.exp(log_tau[missing_labels][valid_keys]) 
            tau[tau!=tau] = 0

            tau = tau.detach().cpu()

            #M-step
            if self.deterministic_EM: #EM only on observed variables
                print('M step only on deterministic labels')
                mu, Sigma = self.M_step(tau[~missing_labels], Z[~missing_labels], self.mu)
                self.mu[mu==mu] = mu[mu==mu] #replace when different from nan
                self.Sigma[Sigma==Sigma] = Sigma[Sigma==Sigma] #replace when different from nan

            elif self.tempered_EM:

                mu_0, Sigma_0 = self.M_step(tau[missing_labels], Z[missing_labels])
                mu_1, Sigma_1 = self.M_step(tau[~missing_labels], Z[~missing_labels])

                missing_ratio = len(missing_labels)/len(tau)
                t0, t1 = self.temperature*(1-missing_ratio), (1 - self.temperature)*missing_ratio

                mu = torch.mean(torch.stack((mu_0*t0,mu_1*t1),axis=2),axis=2)
                Sigma = torch.mean(torch.stack((Sigma_0*t0,Sigma_1*t1),axis=2),axis=2)

                self.mu[mu==mu] = mu[mu==mu] #replace when different from nan
                self.Sigma[Sigma==Sigma] = Sigma[Sigma==Sigma] #replace when different from nan

            else:
                mu, Sigma = self.M_step(tau, Z)
                self.mu[mu==mu] = mu[mu==mu] #replace when different from nan
                self.Sigma[Sigma==Sigma] = Sigma[Sigma==Sigma] #replace when different from nan

            #set to device
            self.tau = tau.to(self.device)
            self.mu = self.mu.to(self.device)
            self.Sigma = self.Sigma.to(self.device)

        self.hist['tau'] += [self.tau]

        if self.plot==True:
            M = 50000
            n = max(1,Z.shape[0]//M)
            X = Z.detach().numpy()
            pca = PCA(n_components=2)
            pca.fit(X)
            X = X@pca.components_.T
            fig, ax = plt.subplots(3,1,figsize=(18,8))
            sample = (self.labels[:,self.K:].sum(axis=-1)==0) #sample on non missing data points
            ax[0].scatter(X[sample,0],X[sample,1],c=torch.where(self.labels==1)[1][sample].detach(),alpha=0.8)
            color = np.concatenate((tau.detach().numpy(),np.zeros(len(tau)).reshape(-1,1)),axis=1)
            tau_sum = tau.mean(axis=0).detach().numpy()
            tau_sum = tau_sum/max(tau_sum)
            print('tau sum',tau_sum)


            for i in range(self.K):
                mu = (self.mu[i]@pca.components_.T).detach().numpy()
                Sigma = (pca.components_@torch.diag(self.Sigma[i]).detach().numpy()@pca.components_.T)
                ax[0] = self.draw_95_ellipse(mu, Sigma, alpha = tau_sum[i], ax=ax[0])

            step = [0] + list(np.where(self.tempo.detach().numpy()==1)[0])
            for i in range(30):
                y = self.labels[step[i]:step[i+1],self.i_i50].detach().numpy()
                if len(y)>0:
                    y = GF1D(y,10)
                    if max(y)>0:
                        y = y/max(y)

                    ax[1].plot(X[step[i]:step[i+1],0],X[step[i]:step[i+1],1],alpha=0.3)
                    ax[1].scatter(X[step[i]:step[i+1],0],X[step[i]:step[i+1],1],c=y*50,alpha=0.8)

            img = tau[::1][:200].detach().numpy().T
            img[0] = (self.tempo==1)[:200]
            ax[2].imshow(img)

            fig.suptitle(f'Scatterplot on 2d PCA at epoch {self.epoch}')
            fig.savefig(f'plot{self.epoch:03d}.png')
        
        self.set_vector_field()
        self.Z = None
        self.hist['mu'] = torch.vstack([self.hist['mu'],self.mu])
        self.hist['Sigma'] = torch.vstack([self.hist['Sigma'],self.Sigma])
        self.hist['beta'] += [self.beta]

    def M_step(self, tau, Z, mu_prior=None):

        tau_sum = tau[:,:,None].sum(axis=0).detach().cpu()
        #mu = (tau[:,:,None]*Z[:,None,:].detach().cpu()).sum(axis=0).detach().cpu()/tau_sum
        mu = tau.T@Z.detach().cpu() / tau.sum(axis=0).reshape(-1,1).repeat(1,Z.shape[-1]).detach().cpu()
        Sigma = (tau[:,:,None] * (Z[:,None,:].detach().cpu()-mu[None,:,:].detach().cpu())**2).sum(axis=0).detach().cpu()/tau_sum + 0.00001
        #mu, Sigma = self.mu, self.Sigma
        return mu, Sigma

## compute vector field

    def set_vector_field(self):
        """
        Vector field when we have a defined structure like 1095 * n patient obs
        """
        z = self.Z
        z = z.reshape(self.N,-1,z.shape[-1])
        delta_z = z[1:] - z[:-1]
        self.v = (z[1:] - z[:-1]).reshape(-1,z.shape[-1])
        self.v_start = z[:-1].reshape(-1,z.shape[-1])

    def set_vector_field(self):
        """
        Vector field when we have tempo describing how close we are to the end of a series of observations
        """
        z = self.Z
        self.v = (z[1:] - z[:-1])[self.tempo[:-1]!=1]
        self.v_start = z[:-1][self.tempo[:-1]!=1]

    def vector_kernel_dist(self, z1, z2):
        """
        Returns the kernel distance between z1 and z2 for vector field estimation
        """
        if z1.shape[0]==1:
            z1 = z1.repeat(z2.shape[0])
        return torch.exp(-torch.mean( (z1[:,None,:] - z2[None,:,:])**2 / self.S , axis = -1))

    def compute_vector(self,z):
        v = self.vector_kernel_dist(z,self.v_start) @ self.v / self.vector_kernel_dist(z,self.v_start).sum(axis=1)[:,None]
        return v

## plot intermediate EM steps

    def draw_95_ellipse(self, mu, Sigma, c="black", alpha=1, ax=None):
        if len(Sigma.shape) == 1:
            Sigma = torch.diag(Sigma)
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        axa = np.sqrt(eigenvalues[0] * 5.991)
        axb = np.sqrt(eigenvalues[1] * 5.991)
        va = eigenvectors[:, 0]
        vb = eigenvectors[:, 1]
        x = np.linspace(0, 2*np.pi, 100)
        trace = np.array([np.cos(e)*va*axa + np.sin(e)*vb*axb + mu for e in x])
        ax.plot(trace[:,0], trace[:,1], c=c, alpha=alpha, linewidth=3)

        return ax 
   
    @classmethod
    def _load_model_config_from_folder(cls, dir_path):
        file_list = os.listdir(dir_path)

        if "model_config.json" not in file_list:
            raise FileNotFoundError(
                f"Missing model config file ('model_config.json') in"
                f"{dir_path}... Cannot perform model building."
            )

        path_to_model_config = os.path.join(dir_path, "model_config.json")
        model_config = vEMAEConfig.from_json_file(path_to_model_config)

        return model_config


    
