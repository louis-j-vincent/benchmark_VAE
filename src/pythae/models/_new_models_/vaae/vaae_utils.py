import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.base_utils import ModelOutput
from .vaae_config import VAAEConfig
import numpy as np

#plotter
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
import pandas as pd

class Plotter(nn.Module):
    def __init__(self, model):

        nn.Module.__init__(self)

    def plot(self, model):

        if True:

            #x_true = torch.nan_to_num(model.x.clone().detach(),nan=0)
            #x_tronc = model.DataMasker.corrupt()
            x_true = self.x_true.clone()
            x_corr = self.x_corr.clone()

            # get latent rpz
            out = model.encoder(x_corr)
            mu, var = out.embedding, out.covariance
            std_norm = var**.5
            z_min, z, z_max = mu - 2*std_norm, mu, mu + 2*std_norm

            # get output
            x_recon = model.decoder(z).reconstruction.reshape(x_true.shape).detach()
            x_recon_min = model.decoder(z_min).reconstruction.reshape(x_true.shape).detach()
            x_recon_max = model.decoder(z_max).reconstruction.reshape(x_true.shape).detach()

        x_true[x_true==0] = float('nan')
        x_corr[x_corr==0] = float('nan')


        n_plots_per_axis = 3
        n_feats = x_true.shape[1]
        x_axis = np.arange(len(x_true))
        n_plots = int(np.ceil(n_feats / n_plots_per_axis))
        #if n_plots == 1:
        #    n_plots = 2
        fig, ax = plt.subplots(n_plots,figsize=(18,5*n_plots))
        colors = ['r','g','b','y','c']
        for i in range(n_plots):
            plt.gca().set_prop_cycle(None)
            
            range_j = range(n_plots_per_axis*i,min(n_plots_per_axis*(i+1),n_feats))
            for j in range_j:
                ax[i].plot(x_corr[:,j], linewidth = 2, c = colors[j%n_plots_per_axis])
            plt.gca().set_prop_cycle(None)
            for j in range_j:
                ax[i].plot(x_corr[:,j], 'o', c = colors[j%n_plots_per_axis])
            plt.gca().set_prop_cycle(None)
            for j in range_j:
                ax[i].plot(x_true[:,j], '+', c = colors[j%n_plots_per_axis])
            plt.gca().set_prop_cycle(None)
            for j in range_j:
                ax[i].plot(x_recon[:,j],':', c = colors[j%n_plots_per_axis])
            plt.gca().set_prop_cycle(None)
            for j in range_j:
                ax[i].fill_between(x_axis, x_recon_min[:,j], x_recon_max[:,j], alpha = .5, color = colors[j%n_plots_per_axis])
            ax[i].set_ylim([-4,4])
            ax[i].hlines([0],[0],[len(x_axis)])

        fig.savefig(f'plots/plot_{model.epoch}.png')
        plt.clf()
        plt.close()


    def scatter_plot(self, model):

        Z = model.Z.detach().numpy()
        Z_pca = Z@self.pca.T

        M = 200
        start = time.time()

        fig, ax = plt.subplots(3,1,figsize=(10,6))

        for k in range(self.K):
            # Sample 500 points max from each cluster
            labels_k = torch.where(model.labels[:,k]==1)[0].detach()
            N = max(1,len(labels_k) // M)
            sample = labels_k[::N]

            # plot on 2 axes
            ax[0].scatter(Z_pca[:,0][sample],
                            Z_pca[:,1][sample],
                            c=self.colors[k%self.K],
                            alpha=0.2)
            ax[1].scatter(Z[:,0][sample],
                            Z[:,1][sample],
                            c=self.colors[k%self.K],
                            alpha=0.2)

            mu = (model.mu[k]@self.pca.T).detach().numpy()
            Sigma = (self.pca@torch.diag(model.Sigma[k]).detach().numpy()@self.pca.T)
            print('mu Sigma in plot step', mu, Sigma)
            #print(self.mu)
            ax[0] = self.draw_95_ellipse(mu, Sigma, alpha = min(1,0.9), ax=ax[0], c=self.colors[k%self.K])


            # draw N_Gauss associated projected ellipses for each label k
            if False:
                pi_mean = model.pi.mean(axis=0)
                pi_mean /= (pi_mean @ self.pi_to_tau @ self.pi_to_tau.T)

                for j in torch.where(model.pi_to_tau.T[k]==1)[0].item():

                    alpha = pi_mean[j].clamp(0.1,1).item() #importance of sub cluster
                    mu = (self.mu[j]@self.pca.T).detach().numpy()
                    Sigma = (self.pca@torch.diag(self.Sigma[j]).detach().numpy()@self.pca.T)
                    ax[0] = self.draw_95_ellipse(mu, 
                                                Sigma, 
                                                alpha = alpha, 
                                                ax=ax[0], 
                                                c=self.colors[k%self.K])
                    ax[1] = self.draw_95_ellipse(self.mu[j,:2].detach().numpy(), 
                                                torch.diag(self.Sigma[j,:2]).detach().numpy(), 
                                                alpha = alpha, 
                                                ax=ax[1], 
                                                c=self.colors[k%self.K])

        try:
            img = model.tau[:300].detach().numpy().T
            img[0] = (model.tempo==1)[:300]
            ax[2].imshow(img)
        except:
            pass

        fig.suptitle(f'Scatterplot on 2d PCA at epoch {model.epoch}')
        fig.savefig(f'plots/plot{model.epoch:03d}.png')

    def plot_losses(self, losses, epoch, columns):
        
        fig, ax = plt.subplots(figsize=(10,6))

        df_losses = pd.DataFrame(losses, 
                                    columns = columns)

        df_losses.rolling(3).mean().plot(logy=True,ax=ax) #.drop('temporal',axis=1)
        ax.set_ylim([10**-5,10**4])

        fig.suptitle(f'Losses at epoch {epoch}')
        fig.savefig(f'plots/losses.png')

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

class DataMasker(nn.Module):
    def __init__(self, model):

        nn.Module.__init__(self)

        self.p = model.p
        self.n_repeats = model.n_repeats
        self.nan_token = model.nan_token

    def data_augmentation(self, x: torch.Tensor, *args: torch.Tensor):
        """
        Data augmentation classique
        """
        n_repeats = self.n_repeats# if self.training else 1

        X = x.repeat_interleave(n_repeats,dim=0)
        XV = self.corrupt(X) 

        args = list(args)
        for i in range(len(args)):
            args[i] = args[i].repeat_interleave(n_repeats,dim=0)
     
        return X, XV, *args

    def corrupt(self, X):
        """
        Adds missing values with probability p
        """
        #if variance, then U should be divided then doubled up

        U = (X!=self.nan_token)*1 #get mask on missing values
        p = U*self.p
        V = torch.bernoulli( p )
        V[::self.n_repeats] = 0 #preserve the first element as a copy with no extra missing info
        XV = X.clone()
        XV[V==1] = self.nan_token #add missing value when V mask == 1

        return XV



