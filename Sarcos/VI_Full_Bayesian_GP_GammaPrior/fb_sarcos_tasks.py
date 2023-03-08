
import torch
import numpy as np
import random
from Sarcos.kernel import *




class fb_Sarcos_Task():
    def __init__(self, idx, X, y, Xstar, Ystar,post_mean_etaparms, post_cov_etaparms):
        """
        :param a:
        :param z:
        """
        self.idx = idx

        self.N = 1000
        li = list(range(0, X.size()[0]))
        random.seed(0)
        self.sub_idces = random.sample(li, self.N)
        self.X = X[self.sub_idces]
        self.y = y[self.sub_idces]

        self.Nprime = 100
        lisub = list(range(0, self.X.size()[0]))
        random.seed(0)
        self.subsub_idces = random.sample(lisub, self.Nprime)
        self.Xprime = self.X[self.subsub_idces]



        # Set the testing point which indexes the integrand
        self.xstar = Xstar[self.idx].squeeze().unsqueeze(0)
        self.ystar = Ystar[self.idx].squeeze()

        self.post_mean_etaparms = post_mean_etaparms
        self.post_cov_etaparms = post_cov_etaparms



    def true_integral_val(self):
        return self.ystar.numpy()


    def integrand(self, X):
        eta=X
        theta= eta.exp()
        n = theta.size()[0]
        f_vals = torch.zeros(n,1)
        for i in range(n):
            K_star_Nprime = base_kernel(self.xstar, self.Xprime, theta[i])
            K_Nprime = base_kernel(self.Xprime,self.Xprime,theta[i])
            K_Nprime_N = base_kernel(self.Xprime,self.X,theta[i])
            K_N_Nprime = K_Nprime_N.t()
            sigma = 0.1
            f_vals[i,:] = (K_star_Nprime @ torch.inverse(K_Nprime_N @ K_N_Nprime + sigma**2 * K_Nprime) @ (K_Nprime_N @ self.y)).squeeze()
        return f_vals.numpy()



    def sample_data(self, size=1):
        """
        Sample data from this task.
        :returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        m1 = torch.distributions.multivariate_normal.MultivariateNormal(self.post_mean_etaparms, self.post_cov_etaparms)
        eta = m1.sample((size,))
        f_eta = self.integrand(eta)


        f_eta = torch.tensor(f_eta, dtype=torch.float)

        # X,y
        return eta.float(), f_eta.float()





class fb_Sarcos_Task_Distribution():
    def __init__(self, idces_list,  X, y, Xstar, Ystar, post_mean_etaparms, post_cov_etaparms ):
        self.idces_list = idces_list
        self.X = X
        self.y = y
        self.Xstar = Xstar
        self.Ystar = Ystar
        self.post_mean_etaparms = post_mean_etaparms
        self.post_cov_etaparms = post_cov_etaparms

    def sample_task(self):
        idx = random.sample(self.idces_list, 1)
        return fb_Sarcos_Task(idx,  self.X, self.y, self.Xstar, self.Ystar, self.post_mean_etaparms, self.post_cov_etaparms)





