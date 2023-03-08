import torch
import numpy as np
import random
from Sarcos.kernel import *


class neg_Elbo_VI_Baye_GP(torch.nn.Module):
    def __init__(self, kernel):
        super(neg_Elbo_VI_Baye_GP, self).__init__()
        self.kernel = kernel  # a kernel class

        # Variational parameters -- using full rank q(theta)
        self.mu = torch.nn.Parameter(torch.zeros(2, dtype=torch.float, requires_grad=True))  # 2d tensor, 2 * 1
        self.L = torch.nn.Parameter(torch.eye(2, dtype=torch.float, requires_grad=True) * torch.tensor([0.002,0.002]))  # 2d tensor, 2 * 2

    def log_density_Gaussian(self, x, mu, cov):
        # a helper function to return the log density of N(mu, cov) at one sample x
        __distrib = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        return __distrib.log_prob(x)

    def log_density_mul_Gamma(self,x, a=25., b=25., c=25.,d=25.):
        # x is a 2-by-1 tensor,
        __m1 = torch.distributions.gamma.Gamma(torch.tensor(a, dtype=float), torch.tensor(c, dtype=float))
        lpb_x1 = __m1.log_prob(x[0])
        __m2 = torch.distributions.gamma.Gamma(torch.tensor(b, dtype=float), torch.tensor(d, dtype=float))
        lpb_x2 = __m2.log_prob(x[1])
        lpb_x = lpb_x1 + lpb_x2
        return lpb_x

    def forward(self, X, Y, sample_size_for_elbo=10, inference_on_eta=False):
        X_batch = X
        Y_batch = Y

        elbo = 0.
        for i in range(sample_size_for_elbo):
            # sample from the variational distribution: q(theta)
            self.eta = self.mu + torch.tril(self.L) @ torch.randn(2)

            # use exp-log transformation
            k_XbXb = self.kernel(X_batch, X_batch, self.eta.exp())
            k_Yb = k_XbXb + (0.1 ** 2) * torch.eye(X_batch.size()[0])
            k_Yb.to(dtype=torch.float64)
            if Y_batch.dim() == 1:
                Y_batch = Y_batch.unsqueeze(dim=1)  # ensure Y is a column vector
            distrib = torch.distributions.MultivariateNormal(torch.zeros(Y_batch.size()[0]), covariance_matrix=k_Yb)
            log_mll = distrib.log_prob(Y_batch.squeeze())

            if inference_on_eta:
                # Donot need jacobian adjust as we can specify priors on eta directly rather than on the eta.exp() [which is kernel params theta]
                log_q = self.log_density_Gaussian(self.eta, self.mu,\
                                              torch.tril(self.L) @ torch.tril(self.L).t())
                # Note here is the prior on eta
                log_prior = self.log_density_Gaussian(self.eta, torch.zeros(2),\
                                                  torch.eye(2) * torch.tensor([0.02, 0.02]))

                elbo += log_mll + log_prior - log_q
            else:
                # inference on the theta scale
                # Note here is the prior on theta
                log_prior = self.log_density_mul_Gamma(self.eta.exp(),  a=25., b=25., c=25.,d=25. )
                log_det_jacobian = torch.log(self.eta.squeeze().exp().prod())
                log_q = self.log_density_Gaussian(self.eta, self.mu,\
                                              torch.tril(self.L) @ torch.tril(self.L).t())



                elbo += log_mll + log_prior + log_det_jacobian - log_q


        # MC estimator:
        elbo = elbo / sample_size_for_elbo

        # The goal is to minimise the negative ELBO..
        neg_elbo = -1. * elbo
        return neg_elbo




class train_avi_fullbayesian_GP(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def optim_neg_elbo(self, X_tr, Y_tr, lr, sample_size_for_elbo, epochs, inference_on_eta=False, verbose=False):
        self.neg_elbo = neg_Elbo_VI_Baye_GP(self.kernel)
        optimizer = torch.optim.Adam(self.neg_elbo.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        for i in range(epochs):
            optimizer.zero_grad()
            out = self.neg_elbo(X=X_tr, Y=Y_tr, sample_size_for_elbo=sample_size_for_elbo, inference_on_eta=inference_on_eta)
            out.backward()
            optimizer.step()
            scheduler.step()

            if verbose:
                if i % 10 ==0:
                    print(i + 1, iter, out, self.neg_elbo.mu.detach(), self.neg_elbo.L.detach())



import os
os.getcwd()
X = torch.load('Sarcos/X.pt')
y = torch.load('Sarcos/y.pt')
Xstar = torch.load('Sarcos/Xstar.pt')
Ystar = torch.load('Sarcos/ystar.pt')
X = X.float()
y = y.float()
Xstar = Xstar.float()
Ystar = Ystar.float()

N = 1000
li = list(range(0, X.size()[0]))
random.seed(0)  #
sub_idces = random.sample(li, N)
X = X[sub_idces]
y = y[sub_idces]

# Train
my_gp = train_avi_fullbayesian_GP(rbf_kernel)
my_gp.optim_neg_elbo(X_tr=X, Y_tr=y, lr=2e-3, sample_size_for_elbo = 10, inference_on_eta=False, epochs=300, verbose=True)


# Get results
post_mean_etaparms = my_gp.neg_elbo.mu.detach()
post_cov_etaparms = my_gp.neg_elbo.L.detach() @ my_gp.neg_elbo.L.detach().t()
post_distr_etaparms = torch.distributions.MultivariateNormal(post_mean_etaparms.squeeze(), covariance_matrix=post_cov_etaparms)

# With Gamma prior, we have,
# post_mean_etaparms = torch.tensor([-0.1824,  0.1950])
# post_cov_etaparms = torch.tensor([[ 0.0029, -0.0025], [-0.0025,  0.0065]])
