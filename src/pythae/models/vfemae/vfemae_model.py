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
import time
import pandas as pd


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
        ## PARAMS RELATIVE TO EMAE
        self.delta = 1
        self.beta = 1
        self.gamma = 1
        self.EM_steps = 10
        self.temperature = 0.01
        self.variationnal = True
        self.quantile = torch.tensor(0.5)
        self.plot = False
        self.temp_start = -1
        self.use_missing_labels = False
        self.print = False
        self.Z = torch.tensor([])
        self.labels = torch.tensor([])
        self.tempo = torch.tensor([])
        self.losses = torch.tensor([])
        self.mfgs = torch.tensor([])

        ##init EMAE params
        self.init = True
        self.K = model_config.K #nb of Gaussians
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mu = torch.zeros((self.K,model_config.latent_dim)).to(device)
        print('self.K is ',self.K,self.mu.shape[1])
        #print(min(self.K,self.mu.shape[1]))
        #for k in range(min(self.K,self.mu.shape[1])):
        #    self.mu[k,k] = 1.
        self.Sigma = torch.ones(self.mu.shape).to(device)
        self.cluster_map = torch.eye(self.K)
        self.device = device
        self.alpha = (torch.ones(self.K)/self.K).to(device) #prior probabilities for each gaussian 
        self.hist = {'mu':self.mu, 'Sigma':self.Sigma, 'beta':[self.beta], 'tau':[]}
        self.recon_loss, self.ll_loss = None, None
        self.epoch = 0
        self.i_i50 = 18
        self.deterministic_EM = False

        ##Rieman
        self.M = []
        self.centroids = []
        self.S = 1
        
        config = vfEMAEConfig(input_dim=(self.latent_dim+1,1),
                                latent_dim=self.latent_dim)
        self.distribution_shift = Encoder_AE_MLP(config) #nn that will account for distribution shifts

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
        labels, tempo, mfgs = inputs["labels"][:,0].to(torch.int64), inputs["labels"][:,1].to(torch.float64), inputs["labels"][:,2].to(torch.int64)
        y_missing = F.one_hot(labels,num_classes=self.K*3).float().to(self.device) #3 to overshoot if there are many classes
        mfgs = F.one_hot(mfgs,num_classes=4).to(self.device)
        y = y_missing[:,:self.K]

        z = self.encoder(x,mfgs).embedding

        ## update args
        self.Z = torch.cat((self.Z, z),0)
        self.labels = torch.cat((self.labels, y_missing),0)
        self.tempo = torch.cat((self.tempo, tempo),0)
        self.mfgs = torch.cat((self.mfgs, mfgs),0)

        LLloss, classif_loss = self.likelihood_loss(z,y_missing) #Log likelihood loss on all data 
        
        ## one step of EMAE variationnal sampling
        z_var = self.variationnal_sampling(y,z)

        ## NORMAL vAE step
        X, Z = x.repeat_interleave(n_repeats,dim=0), z_var.repeat_interleave(n_repeats,dim=0)
        Mfgs = mfgs.repeat_interleave(n_repeats, dim=0)
        XV = self.corrupt(X) #corrupt by adding missing values

        ZV_mu = self.encoder(XV,Mfgs).embedding #encoding of xU
        XV_hat = self.decoder(ZV_mu,Mfgs)["reconstruction"].reshape(ZV_mu.shape[0],x.shape[-1])

        sample = np.random.choice(np.arange(len(z)),size=200,replace=True)
        hsic_loss = self.HSIC(z[sample], mfgs[sample], s_x=1, s_y=1)

        #recon_loss = self.loss_function(XV_hat[(X!=-10)], X[(X!=-10)]) / torch.std(X[(X!=-10)]) * 10
        recon_loss = self.mse_std(XV_hat, X)

        temporal_loss = self.temporal_loss(ZV_mu,X,tempo,n_repeats)

        loss = recon_loss + temporal_loss * self.gamma

        if self.temperature > self.temp_start:
            loss += (LLloss + classif_loss)*self.beta#*self.temperature 

        reg_loss = self.loss_function(ZV_mu,ZV_mu*0.)*0.01

        loss += hsic_loss*self.delta
        loss += reg_loss
        loss += torch.abs(self.Sigma[self.i_i50]).mean()
        loss = loss.mean()

        #self.centroids.append(z.detach().to(self.device))
        #self.M.append( (torch.eye(z.shape[-1]).repeat(z.shape[0],1,1) * tempo[:,None,None]).to(self.device) ) #id matrices whose size increases with time to hosp

        losses_tensor = torch.tensor([self.epoch, 
                                    recon_loss.item(),
                                    LLloss.item(),
                                    classif_loss.item(),
                                    temporal_loss.item(),
                                    reg_loss.item(),
                                    hsic_loss.item(),
                                    loss.item()]).reshape(1,-1)
        self.losses = torch.cat((self.losses, losses_tensor),0)

        ## print to observe evolution of different losses
        if self.print:
            with open('log.txt', 'a') as f:
                str1 = f'Losses: e{self.epoch:.4f} -recon {recon_loss.item():.4f} -LLoss {LLloss.item():.4f} classif {classif_loss.item():.4f} ' 
                str2 = f'-temporal {temporal_loss.item():.4f} latent {reg_loss.item():.4f} HSIC {hsic_loss.item():.4f} total {loss.item():.4f}\n'
                f.write(str1)
                f.write(str2)
        
        output = ModelOutput(loss=loss, recon_x=XV_hat, z=Z)

        return output

    def variationnal_sampling(self,y,z):
        if self.variationnal: #do a VAE-like step by sampling on a gaussian centered on z and "contained" inside the class gaussian
            ratio = ((self.quantile)/(1.96 + torch.abs(z - y@self.mu)))**2
            sigma_small_max = torch.fmax((y@self.Sigma**0.5)*ratio,torch.tensor(0.001)) 
            return z + torch.normal(torch.zeros(z.shape).to(self.device),sigma_small_max).to(self.device)
        else:
            return z

    def temporal_loss_structured(self,Z,X,tempo,n_repeats):
        """
        Temporal loss

        # we divide (aka multiply here) by delta_U sum to get mean of delta_x * delta_U only where delta_U==1
            #print(temporal_loss)
        """
        delta_Z = torch.abs(Z[1:] - Z[:-1])
        delta_X = torch.abs(X[1:] - X[:-1])**2 + 0.0001
        U = (X==-10)
        delta_U = (1 - U[1:]*U[:-1]*1) + 0.0001
        return torch.abs( delta_Z.mean(axis=-1) * delta_U.sum(axis=-1) / (delta_X * delta_U).sum(axis=-1) ).mean() * (delta_X * delta_U).mean()
        
    def temporal_loss(self,Z,X,tempo,n_repeats):
        """
        Temporal loss

        # we divide (aka multiply here) by delta_U sum to get mean of delta_x * delta_U only where delta_U==1
            #print(temporal_loss)
        """
        delta_Z = torch.abs(Z[1:] - Z[:-1])
        TEMPO = tempo.repeat_interleave(n_repeats,dim=0)
        return torch.abs(delta_Z[TEMPO[:-1]!=1]).mean()

    def corrupt(self, X):
        """
        Adds missing values with probability p
        """
        #if variance, then U should be divided then doubled up

        U = (X!=-10)*1 #get mask on missing values
        U_half = U[:,:U.shape[1]//2]
        V = torch.bernoulli( U_half*self.p )
        V = V.repeat(1,2)
        XV = X.clone()
        XV[V==1] = -10 #add missing value when V mask == 1

        return XV

    def loss_function(self, recon_x, x):

        if recon_x.shape != x.shape:
            recon_x = torch.squeeze(recon_x, 1) #if dimension added

        MSE = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)
        return MSE.mean(dim=0)

    def mse_std(self, recon_x, x):
        #std =  torch.nanmean( (x - torch.nanmean(x,axis=0))**2, axis=0 ).repeat(x.shape[0],1)
        n_nans = len(x) - x.isnan().sum(axis=0)
        mu = torch.nansum(x,axis=0) / n_nans
        std =  torch.sqrt( torch.nansum( (x - mu)**2, axis=0 ) / n_nans )
        std = std.repeat(x.shape[0],1)

        #return torch.mean( (recon_x[x!=-10] - x[x!=-10])**2 / (std[x!=-10] + 1e-5) )
        return torch.mean( (recon_x[(x!=-10)] - x[(x!=-10)])**2 / (std[(x!=-10)] + 1e-5) )
 
    def HSIC(self, x, y, s_x=1, s_y=1):
        m,_ = x.shape #batch size
        K = GaussianKernelMatrix(x,s_x)
        L = CategoricalKernelMatrix(y,s_y)
        self.L_mat, self.K_mat = L, K
        H = torch.eye(m) - 1.0/m * torch.ones((m,m))
        H = H.float().to(self.device)
        HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        return HSIC

## compute likelihood loss

    def E_step(self, Z):
        """
        Given Z, computes and returns tau the matrix of probabilities of belonging to each gaussian
        """

        Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
        Sigma = self.Sigma[None,:,:] + 0.0001

        #E-step
        N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma) )
        log_tau = torch.log(self.alpha[None,:] + 1e-4) + N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
        log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None])#.detach().cpu()
        self.log_tau = log_tau
        tau = torch.exp(log_tau)

        return tau , Y, Sigma

    def log_prob(self,Y,Z,Sigma,tau):
        """
        Given Z and tau, returns the log prob
        """
        N_log_prob = torch.minimum(-0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma)),torch.tensor(0) ) # -0.5*(y**2/sigma +)
        N_prob = (torch.exp(N_log_prob) * tau[:,:self.K,None]).sum(axis=1) #only use the kth gaussian 
        prob = torch.mean(N_prob[N_prob==N_prob]) #nanmean walkaround
        #prob *= (torch.exp(torch.minimum(-0.5* ( Z**2 + torch.log(2*torch.tensor(np.pi))),torch.tensor(0) )).mean())**0.1

        #if tau_prior is not None:
        #    prob -= ((tau.mean(axis=0) - self.alpha)**2 / 2 ).sum()

        return prob

    def log_prob_(self,Y,Sigma,tau):
        """
        Given Z and tau, returns the log prob
        """
        N_log_prob = torch.minimum(-0.5* ( Y**2/Sigma + torch.log(2*np.pi*Sigma)),torch.tensor(0) )#.sum(axis=0)
        N_prob = torch.log(torch.exp(N_log_prob) * tau[:,:self.K,None]).sum(axis=1) #only use the kth gaussian 
        #prob = torch.mean(N_prob[N_prob==N_prob]) #nanmean walkaround

        #if tau_prior is not None:
        #    prob -= ((tau.mean(axis=0) - self.alpha)**2 / 2 ).sum()

        return N_prob

    def M_step(self, tau, Z, mu_prior=None):

        tau_sum = tau[:,:,None].sum(axis=0)
        tau_sum[tau_sum==0] = 1
        mu = tau.T@Z / tau_sum
        Sigma = (tau[:,:,None] * (Z[:,None,:]-mu[None,:,:])**2).sum(axis=0)/tau_sum + 0.00001
        #mu[self.i_i50] = 0
        #Sigma[self.i_i50] /= 2
        return mu.detach().cpu(), Sigma.detach().cpu()

    def likelihood_loss(self, Z, y, ll_with_missing_labels = True):
        """
        Compute likelihood of gaussian Mixture Model by:
        - computing tau (if we are using missing labels) which is the a posteriori probability for each point of being a part of each of hte K clusters
        - computing the likelihood loss defined as sum_i sum_k tau_i,k * P(z_i sachant mu_k, Sigma_k)
        """

        missing_labels = (y[:,:self.K].sum(axis=1)==0)

        #E-step
        tau, Y, Sigma = self.E_step(Z)

        #get log prob
        prob = self.log_prob(Y,Z,Sigma,tau)

        tau_prior = torch.clone(y[:,:self.K])#.detach().cpu()
        if missing_labels.float().mean()!=1:
            loss_classif = self.loss_function( tau_prior[~missing_labels], tau[~missing_labels] )
        else:
            loss_classif = torch.tensor(0)
        #loss_classif = torch.abs( tau_prior[~missing_labels], tau[~missing_labels] ).sum(axis=1).mean()

        return 1 - prob.mean(), loss_classif

    def likelihood_loss_(self, Z, y, ll_with_missing_labels = True):
        """
        Compute likelihood of gaussian Mixture Model by:
        - computing tau (if we are using missing labels) which is the a posteriori probability for each point of being a part of each of hte K clusters
        - computing the likelihood loss defined as sum_i sum_k tau_i,k * P(z_i sachant mu_k, Sigma_k)
        """

        missing_labels = (y[:,self.K:].sum(axis=1)>0)

        #E-step
        tau, Y, Sigma = self.E_step(Z)

        #get log prob
        log_prob = self.log_prob(Y,Sigma,tau)

        tau_prior = torch.clone(y[:,:self.K])#.detach().cpu()
        loss_classif = self.loss_function( tau_prior[~missing_labels], tau[~missing_labels] )

        return - log_prob.mean(), loss_classif

## update parameters

    def update_parameters(self,Z=None):
        """
        Run self.EM_steps (set to 1) steps of the EM algorithm to estimate parameters 
        """

        labels = self.labels[:,:self.K]
        tau = torch.clone(labels).detach().cpu()
        missing_labels = (labels.sum(axis=1)==0)

        if Z==[] or Z is None:
            Z = self.Z
        if self.init==True:
            self.mu = torch.matmul(labels.T,Z)
            self.alpha = labels.sum(axis=0) / labels.sum()
            self.init=False

        print('Updating parameters')

        Z = Z.to(self.device)

        for i in range(self.EM_steps):

            #E-step
            start = time.time()

            tau_new, Y, Sigma = self.E_step(Z)

            d_time = time.time() - start
            start = time.time()
            print(f'E step done in {d_time}')

            tau_new = tau_new#.detach().cpu()
            tau[missing_labels] = tau_new[missing_labels]
            #M-step
            if self.deterministic_EM:
                self.mu, self.Sigma = self.M_step(tau[~missing_labels], Z[~missing_labels], self.mu)
            else:
                self.mu, self.Sigma = self.M_step(tau, Z, self.mu)

            d_time = time.time() - start
            start = time.time()
            print(f'M step done in {d_time}')
            
            if (self.mu!=self.mu).sum()>0:
                print(stoping)

            #set to device
            self.tau = tau.to(self.device)
            self.mu = self.mu.to(self.device)
            self.Sigma = self.Sigma.to(self.device)

        self.hist['tau'] += [self.tau]

        if self.plot:
            self.plot_step()

            d_time = time.time() - start
            start = time.time()
            print(f'plotting done in {d_time}')     

        self.set_vector_field()
        d_time = time.time() - start
        start = time.time()
        print(f'vector field set done in {d_time}')
        self.Z = torch.tensor([])
        self.labels = torch.tensor([])
        self.tempo = torch.tensor([])
        self.mfgs = torch.tensor([])

        self.hist['mu'] = torch.vstack([self.hist['mu'],self.mu])
        self.hist['Sigma'] = torch.vstack([self.hist['Sigma'],self.Sigma])
        self.hist['beta'] += [self.beta]

    def plot_step(self):

        Z, tau = self.Z.detach().numpy(), self.tau
        sample = (self.labels[:,self.K:].sum(axis=-1)==0) #sample on non missing data points
        mfgs = self.mfgs[sample]
        color = np.concatenate((tau.detach().numpy(),np.zeros(len(tau)).reshape(-1,1)),axis=1)
        tau_sum = tau.mean(axis=0).detach().numpy()
        tau_sum = tau_sum/max(tau_sum)

        pca = PCA(n_components=2)
        #mu_diff = (self.mu[self.i_i50] - self.mu[~self.i_i50].mean(axis=0)).reshape(1,-1)
        #pca.fit(mu_diff)
        pca.fit(Z)

        X = Z@pca.components_.T

        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(3,1,figsize=(10,6))
        # scatter plot for pca on latent space
        ax[0].scatter(X[sample,0],X[sample,1],c=torch.where(self.labels==1)[1][sample].detach(),alpha=0.8, cmap = 'tab10')
        # draw associated projected ellipses
        for i in range(self.K):
            mu = (self.mu[i]@pca.components_.T).detach().numpy()
            Sigma = (pca.components_@torch.diag(self.Sigma[i]).detach().numpy()@pca.components_.T)
            ax[0] = self.draw_95_ellipse(mu, Sigma, alpha = tau_sum[i], ax=ax[0], c=cmap(i))
        # plot 30 first trajectories
        step = [0] + list(np.where(self.tempo.detach().numpy()==1)[0])
        for i in range(min(10,len(step)-1)):
            y = self.labels[step[i]:step[i+1],self.i_i50].detach().numpy()
            if len(y)>0:
                y = GF1D(y,10)
                if max(y)>0:
                    y = y/max(y)

                ax[1].plot(X[step[i]:step[i+1],0],X[step[i]:step[i+1],1],alpha=0.3)
                ax[1].scatter(X[step[i]:step[i+1],0],X[step[i]:step[i+1],1],c=y,alpha=0.8)
        # plot first 300 steps of probas of each cluster
        img = tau[::1][:300].detach().numpy().T
        img[0] = (self.tempo==1)[:300]
        ax[2].imshow(img)

        fig.suptitle(f'Scatterplot on 2d PCA at epoch {self.epoch}')
        fig.savefig(f'plot{self.epoch:03d}.png')

        fig, ax = plt.subplots(2,1,figsize=(10,6))

        x = self.X[:200].detach().clone().float()
        z = self.encoder(x,self.mfgs[:200]).embedding.detach().float()
        x_recon = self.decoder(z,self.mfgs[:200]).reconstruction.detach().reshape(x.shape[0],x.shape[1])

        x_recon[x==-10] = float('nan')
        x[x==-10] = float('nan')

        ax[0].plot(x[:,:20])
        plt.gca().set_prop_cycle(None)
        ax[0].plot(x_recon[:,:20],':')
        ax[1].plot(z)

        fig.suptitle(f'X and its reconstruction at epoch {self.epoch}')
        fig.savefig(f'recon{self.epoch:03d}.png')

        fig, ax = plt.subplots(1,2,figsize=(10,6))


        ax[0].imshow(self.L_mat.detach())
        ax[1].imshow(self.K_mat.detach())

        fig.suptitle(f'Correlation matrices of z and y {self.epoch}')
        fig.savefig(f'correl{self.epoch:03d}.png')

        fig, ax = plt.subplots(figsize=(10,6))

        df_losses = pd.DataFrame(self.losses.detach(), 
                                    columns = ['epoch','recon','Ll','classif','temporal','reg','hsic','total'])

        df_losses.rolling(5).mean().plot(logy=True,ax=ax)


        fig.suptitle(f'Losses at epoch {self.epoch}')
        fig.savefig(f'losses.png')

        plt.clf()
       
## compute vector field

    def set_vector_field_structured(self):
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
        return v, self.vector_kernel_dist(z,self.v_start)

## compute Riemannian metric

    def G_inv(self,z):

        self.T = 1 #added
        self.lbd = 0

        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        #print(self.centroids_tens.unsqueeze(0), z.unsqueeze(1))
        
        return (
            self.M_tens.unsqueeze(0)
            * torch.exp(
                -torch.norm(
                    self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                )
                ** 2
                / (self.T ** 2)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        #self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

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
        model_config = vfEMAEConfig.from_json_file(path_to_model_config)

        return model_config

## HSIC

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def CategoricalKernelMatrix(x, sigma=1):
    return torch.mm(x, x.t()).float()



    
