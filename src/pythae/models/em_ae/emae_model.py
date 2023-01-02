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

    ##TO DO : add loss to penalise KNOWN data points that are far from given gaussian (with tau)

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
        self.K = model_config.K #nb of Gaussians
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.mu = torch.rand((self.K,model_config.latent_dim)).to(device)
        self.mu = torch.zeros((self.K,model_config.latent_dim)).to(device)
        for k in range(self.K):
            self.mu[k,k] = 1.
        self.init = True
        self.Sigma = torch.ones(self.mu.shape).to(device)
        self.quantile = torch.tensor(0.5)
        self.device = device
        self.alpha = (torch.ones(self.K)/self.K).to(device) #p probabilities for each gaussian 
        self.hist = {'mu':self.mu, 'Sigma':self.Sigma, 'beta':[self.beta]}
        self.plot = False
        self.recon_loss, self.ll_loss = None, None
        self.temp_start = 0
        self.infer = False
        self.print_tau = True
        self.use_missing_labels = False
        self.EM_steps = 10
        self.gamma = 1

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded
        Args:
            inputs (BaseDataset): An instance of pythae's datasets
        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        y_missing = F.one_hot(inputs["labels"].to(torch.int64),num_classes=self.K+1).float().to(self.device)
        y = y_missing[:,:self.K]
        #print(y_missing.shape, 'y missingshape')
        #print(x[0])
        #print(y_missing[0:5])
        #print('Missing')

        z = self.encoder(x).embedding
        if self.variationnal:
            ratio = ((self.quantile)/(1.96 + torch.abs(z - y@self.mu)))**2
            sigma_small_max = torch.maximum((y@self.Sigma**0.5)*ratio,torch.tensor(0.001))
            z_var = z + torch.normal(torch.zeros(z.shape).to(self.device),sigma_small_max).to(self.device)
        else:
            z_var = z

        if self.Z is None:
            self.Z = z
            self.labels = y_missing
        else:
            self.Z = torch.cat((self.Z, z),0)
            self.labels = torch.cat((self.labels, y_missing),0)
        recon_x = self.decoder(z_var)["reconstruction"]

        loss, recon_loss, embedding_loss = self.loss_function(recon_x, x, z)
    
        if self.beta>0 and self.temperature > self.temp_start:
            LLloss, sep_loss = self.likelihood_loss(z,y)
            loss = recon_loss + LLloss*self.beta
            print(recon_loss.item(), embedding_loss.item(), LLloss.item(),loss.item())
            self.ratio = (LLloss/recon_loss).detach().cpu().numpy().item()
        else:
            loss = recon_loss
            self.ratio = self.beta

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
        tau = torch.clone(y).detach().cpu()
        missing_labels = torch.where(y[:,-1]==1)[0].detach().cpu()
        Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
        Sigma = self.Sigma[None,:,:] 
        N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma) )#.detach().cpu()
        log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
        log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None]).detach().cpu()
        tau[missing_labels] = torch.exp(log_tau[missing_labels])#
        tau = tau.to(self.device) 

        #get log prob
        N_log_prob = torch.minimum(-0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma)),torch.tensor(0) )#.sum(axis=0)
        N_prob = (torch.exp(N_log_prob) * tau[:,:self.K,None]).sum(axis=1) #only use the kth gaussian 
        print(N_prob.mean().item(),(torch.exp(N_log_prob)[~missing_labels] * tau[~missing_labels,:self.K,None]).sum(axis=1).mean().item(),'mean on missing and non missing')           
        prob = N_prob.mean()
        separation_prob = N_prob.prod() #prod on K gaussians

        return 1 - prob.mean(), separation_prob

    def update_parameters(self,Z=None):
        labels = self.labels[:,:self.K]
        if Z is None:
            Z = self.Z
        if self.init==True:
            self.mu = torch.matmul(labels.T,Z)
            self.alpha = labels.mean(axis=0) + (1/self.K)*(self.labels[:,-1].mean())
            self.init=False
        print('Updating parameters')
        for i in range(self.EM_steps):

            #E-step
            tau = torch.clone(labels).detach().cpu()
            #print(tau.mean())
            ## add this to complete missing values
            if self.infer == True:
                missing_labels = torch.where(self.labels[:,-1]==1)[0].detach().cpu()
                Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
                Sigma = self.Sigma[None,:,:] 
                N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma) )#.detach().cpu()
                log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
                log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None]).detach().cpu()
                tau[missing_labels] = torch.exp(log_tau[missing_labels]) 

                #plt.scatter(torch.exp(log_tau[~missing_labels]).detach().cpu(),tau[~missing_labels].detach().cpu());
                #plt.show();
                print('Mean diff between tau and truth')
                print((torch.exp(log_tau[~missing_labels]).detach().cpu() - tau[~missing_labels].detach().cpu()).mean().item())
                tau = tau.detach().cpu()
                if self.print_tau:
                    print(tau.mean(axis=0))
                    print(tau.mean(axis=1))
                    print(tau[missing_labels].mean(axis=0))
                    print(tau[missing_labels].mean(axis=1))

            # M-step
            if self.use_missing_labels:
                tau_sum = tau[:,:,None].sum(axis=0).detach().cpu()
                self.mu = (tau[:,:,None]*Z[:,None,:].detach().cpu()).sum(axis=0).detach().cpu()/tau_sum
                self.Sigma = (tau[:,:,None] * (Z[:,None,:].detach().cpu()-self.mu[None,:,:].detach().cpu())**2).sum(axis=0).detach().cpu()/tau_sum
            else:
                tau_sum = tau[~missing_labels,:,None].sum(axis=0).detach().cpu()
                self.mu = (tau[~missing_labels,:,None]*Z[~missing_labels,None,:].detach().cpu()).sum(axis=0).detach().cpu()/tau_sum
                self.Sigma = (tau[~missing_labels,:,None] * (Z[~missing_labels,None,:].detach().cpu()-self.mu[None,:,:].detach().cpu())**2).sum(axis=0).detach().cpu()/tau_sum
        
            self.tau = tau.to(self.device)
            self.mu = self.mu.to(self.device)
            self.Sigma = self.Sigma.to(self.device)

        #ratio = self.recon_loss/self.ll_loss #*self.temperature
        if self.ratio > 1 and self.beta < 1:
            self.beta = self.beta * (1 + (self.epoch+1)**(-0.5))
        elif self.ratio < 1:
            self.beta = self.beta * (1 - (self.epoch+1)**(-0.5))
        #print(f'beta is now {self.beta}, ratio was {ratio} with temp {self.temperature}')
        #self.beta = self.ratio
        print(f'beta is now {self.beta}')

        if self.plot==True:
            X = Z.detach().numpy()
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            fig, ax = plt.subplots(1,1,figsize=(18,8))
            ax.scatter(X[:,0],X[:,1],c=torch.where(self.labels==1)[1].detach())
            for i in range(self.K):
                mu = (self.mu[i]@pca.components_.T).detach().numpy()
                Sigma = (pca.components_@torch.diag(self.Sigma[i]).detach().numpy()@pca.components_.T)
                ax = self.draw_95_ellipse(mu, Sigma, alpha = float(self.alpha[i].detach()), ax=ax)
            fig.savefig(f'plot{self.epoch}.png')
        self.Z = None
        self.hist['mu'] = torch.vstack([self.hist['mu'],self.mu])
        self.hist['Sigma'] = torch.vstack([self.hist['Sigma'],self.Sigma])
        self.hist['beta'] += [self.beta]

    def detach_parameters(self):
        self.Z = self.Z.detach().clone()
        self.mu = self.mu.detach()
        self.Sigma = self.Sigma.detach()
        self.alpha = self.alpha.detach()

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

#new functions for debugging

    def forward(self, inputs: BaseDataset, **kwargs) -> ModelOutput:
        """The input data is encoded and decoded
        Args:
            inputs (BaseDataset): An instance of pythae's datasets
        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters
        """

        x = inputs["data"]
        y_missing = F.one_hot(inputs["labels"].to(torch.int64),num_classes=self.K*2).float().to(self.device)
        y = y_missing[:,:self.K]

        z = self.encoder(x).embedding
        if self.variationnal:
            ratio = ((self.quantile)/(1.96 + torch.abs(z - y@self.mu)))**2
            sigma_small_max = torch.maximum((y@self.Sigma**0.5)*ratio,torch.tensor(0.001))
            z_var = z + torch.normal(torch.zeros(z.shape).to(self.device),sigma_small_max).to(self.device)
        else:
            z_var = z

        if self.Z is None:
            self.Z = z
            self.labels = y_missing
        else:
            self.Z = torch.cat((self.Z, z),0)
            self.labels = torch.cat((self.labels, y_missing),0)
        recon_x = self.decoder(z_var)["reconstruction"]

        loss, recon_loss, embedding_loss = self.loss_function(recon_x, x, z)
    
        if self.beta>0 and self.temperature > self.temp_start:
            LLloss, sep_loss = self.likelihood_loss(z,y_missing)
            LLloss_true, sep_loss_true = self.likelihood_loss(z,y_missing,ll_with_missing_labels=False)

            loss = recon_loss + (LLloss*self.temperature  + (1-self.temperature)*LLloss_true)*self.beta*self.temperature
            var_loss = self.inter_outer_variance_loss()
            loss += var_loss*self.gamma
            print(recon_loss.item(), embedding_loss.item(), LLloss.item(),var_loss.item(),loss.item())
            self.ratio = (LLloss/recon_loss).detach().cpu().numpy().item()
            #self.ratio = 1
        else:
            loss = recon_loss
            self.ratio = self.beta

        output = ModelOutput(
            loss=recon_loss + LLloss*self.temperature,
            recon_loss=recon_loss,
            embedding_loss=embedding_loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def inter_outer_variance_loss(self):
        """
        Loss to minimize variance within each cluster, maximize variance between the center of each cluster
        """
        var_per_cluster = self.sigma.mean()
        var_centers = torch.var(self.mu,dim=0).mean()

        return var_per_cluster/var_centers

    def likelihood_loss(self, Z, y, ll_with_missing_labels = True):

        tau = torch.clone(y[:,:self.K]).detach().cpu()
        missing_labels = torch.where(y[:,self.K:].sum(axis=1)>0)[0].detach().cpu()
        if ll_with_missing_labels:
            #E-step
            Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
            Sigma = self.Sigma[None,:,:] 
            N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma) )#.detach().cpu()
            log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
            log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None]).detach().cpu()
            tau[missing_labels] = torch.exp(log_tau[missing_labels])#
        else:
            tau = torch.clone(y[~missing_labels,:self.K]).detach().cpu()
            Y = (Z[~missing_labels,None,:]-self.mu[None,:,:])
            Sigma = self.Sigma[None,:,:] 
        tau = tau.to(self.device) 

        #get log prob
        N_log_prob = torch.minimum(-0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma)),torch.tensor(0) )#.sum(axis=0)
        N_prob = (torch.exp(N_log_prob) * tau[:,:self.K,None]).sum(axis=1) #only use the kth gaussian 
        prob = N_prob.mean()
        separation_prob = N_prob.prod() #prod on K gaussians

        if torch.isnan(prob): ##check
            return torch.zeros(10).to(self.device), torch.tensor(0).to(self.device)
        else:
            return 1 - prob.mean(), separation_prob

    def update_parameters(self,Z=None):
        labels = self.labels[:,:self.K]
        if Z is None:
            Z = self.Z
        if self.init==True:
            self.mu = torch.matmul(labels.T,Z)
            self.alpha = torch.nanmean(labels,axis=0) + (1/self.K)*(self.labels[:,self.K:].sum(axis=1).mean())
            self.init=False
        print('Updating parameters')
        for i in range(self.EM_steps):

            #E-step
            tau = torch.clone(labels).detach().cpu()
            ## add this to complete missing values
            if self.infer == True:
                missing_labels = torch.where(self.labels[:,self.K:].sum(axis=1)>0)[0].detach().cpu()
                Y = (Z[:,None,:]-self.mu[None,:,:]) #shape: n_obs, k_means, d_dims
                Sigma = self.Sigma[None,:,:] 
                N_log_prob = -0.5* ( Y**2/Sigma + torch.log(2*torch.pi*Sigma) )#.detach().cpu()
                log_tau = torch.log(self.alpha+1e-5)+N_log_prob.sum(axis=2) #log [ p(x_i ; z_i = k) p(z_i = k)]
                log_tau = (log_tau - torch.logsumexp(log_tau, axis=1)[:,None]).detach().cpu()
                tau[missing_labels] = torch.exp(log_tau[missing_labels]) 
                tau = tau.detach().cpu()

            if False:

                #M-step
                tau_sum_0 = tau[missing_labels,:,None].sum(axis=0).detach().cpu()
                mu_0 = (tau[missing_labels,:,None]*Z[missing_labels,None,:].detach().cpu()).sum(axis=0).detach().cpu()/tau_sum_0
                Sigma_0 = (tau[missing_labels,:,None] * (Z[missing_labels,None,:].detach().cpu()-self.mu[None,:,:].detach().cpu())**2).sum(axis=0).detach().cpu()/tau_sum_0
        
                tau_sum_1 = tau[~missing_labels,:,None].sum(axis=0).detach().cpu()
                mu_1 = (tau[~missing_labels,:,None]*Z[~missing_labels,None,:].detach().cpu()).sum(axis=0).detach().cpu()/tau_sum_1
                Sigma_1 = (tau[~missing_labels,:,None] * (Z[~missing_labels,None,:].detach().cpu()-self.mu[None,:,:].detach().cpu())**2).sum(axis=0).detach().cpu()/tau_sum_1
        
                missing_ratio = len(missing_labels)/len(tau)
                t0, t1 = self.temperature*(1-missing_ratio), (1 - self.temperature)*missing_ratio
                
                #t0,t1 = 1 - missing_ratio, missing_ratio
                print('mu and sigma means')
                print((mu_0*t0).mean(axis=1), (mu_1*t1).mean(axis=1))
                print((Sigma_0*t0).mean(axis=1), (Sigma_1*t1).mean(axis=1))

                self.mu = torch.nanmean(torch.stack((mu_0*t0,mu_1*t1),axis=2),axis=2)
                self.Sigma = torch.nanmean(torch.stack((Sigma_0*t0,Sigma_1*t1),axis=2),axis=2)

            else:

                #M-step
                tau_sum = tau[:,:,None].sum(axis=0).detach().cpu()
                self.mu = (tau[:,:,None]*Z[:,None,:].detach().cpu()).sum(axis=0).detach().cpu()/tau_sum
                self.Sigma = (tau[:,:,None] * (Z[:,None,:].detach().cpu()-self.mu[None,:,:].detach().cpu())**2).sum(axis=0).detach().cpu()/tau_sum
        
            #set to device
            self.tau = tau.to(self.device)
            self.mu = self.mu.to(self.device)
            self.Sigma = self.Sigma.to(self.device)

        #if self.ratio > 1 and self.beta < 1:
        #    self.beta = self.beta * (1 + (self.epoch+1)**(-0.5))**(1-self.temperature)
        #elif self.ratio < 1:
        #    self.beta = self.beta * (1 - (self.epoch+1)**(-0.5))**(1-self.temperature)

        #if self.ratio > 1 and self.beta < 1:
        #    self.beta = self.beta * (1.1)**(1-self.temperature)
        #elif self.ratio < 1:
        #    self.beta = self.beta * (1.1)**(-(1-self.temperature))
        print(f'beta is now {self.beta}')

        if self.plot==True:
            X = Z.detach().numpy()
            pca = PCA(n_components=2)
            X = pca.fit_transform(X)
            fig, ax = plt.subplots(1,1,figsize=(18,8))
            ax.scatter(X[:,0],X[:,1],c=torch.where(self.labels==1)[1].detach())
            for i in range(self.K):
                mu = (self.mu[i]@pca.components_.T).detach().numpy()
                Sigma = (pca.components_@torch.diag(self.Sigma[i]).detach().numpy()@pca.components_.T)
                ax = self.draw_95_ellipse(mu, Sigma, alpha = float(self.alpha[i].detach()), ax=ax)
            fig.savefig(f'plot{self.epoch}.png')
        self.Z = None
        self.hist['mu'] = torch.vstack([self.hist['mu'],self.mu])
        self.hist['Sigma'] = torch.vstack([self.hist['Sigma'],self.Sigma])
        self.hist['beta'] += [self.beta]