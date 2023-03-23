### Not used ###

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

## forward

    def AE_step(self, x, mfgs, y_missing):
        """
        One classical AE step
        """
        z = self.encoder(x,mfgs).embedding
        self.Z = torch.cat((self.Z, z),0)

        LLloss, classif_loss = self.likelihood_loss(z,y_missing) #Log likelihood loss on all data 
    
        sample = np.random.choice(np.arange(len(z)),size=200,replace=True)
        hsic_loss = self.HSIC(z[sample], mfgs[sample], s_x=1, s_y=1)

        return LLloss, classif_loss, hsic_loss, z

    def vAE_step(self, x, z, tempo, mfgs):
        """
        Augment with n_repeats 
        """

        n_repeats = self.n_repeats if self.training else 1

        # corrupt by adding missing values
        XV = self.corrupt(X) 
        # augment to get to same size
        X, Z, Mfgs = x.repeat_interleave(n_repeats,dim=0), z.repeat_interleave(n_repeats,dim=0), mfgs.repeat_interleave(n_repeats, dim=0)
        
        # encode / decode
        ZV_mu = self.encoder(XV,Mfgs).embedding #encoding of xU
        XV_hat = self.decoder(ZV_mu,Mfgs)["reconstruction"].reshape(ZV_mu.shape[0],x.shape[-1])

        recon_loss = self.mse_std(XV_hat, X)

        temporal_loss = self.temporal_loss(ZV_mu,X,tempo,n_repeats)

        reg_loss = self.loss_function(ZV_mu,ZV_mu*0.)

        return recon_loss, temporal_loss, reg_loss

## losses 

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

## others

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
     