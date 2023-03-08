
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal

# Under Gamma prior
post_mean_etaparms = torch.tensor([-0.1824,  0.1950])
post_cov_etaparms = torch.tensor([[ 0.0029, -0.0025], [-0.0025,  0.0065]])



prior_data = np.vstack([
        np.random.gamma(25, 0.04, size=(1000000,2))
])


posterior_data = np.vstack([
        np.exp(multivariate_normal(post_mean_etaparms.squeeze().numpy(), post_cov_etaparms.numpy(), size=1000000))
])



fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 6)
axes[0].set_title(r'Prior of $x$', fontsize=22)
axes[0].hist2d(prior_data[:, 0], prior_data[:, 1],density=True, bins=100)
axes[0].set_xlabel(r'$x_1$', fontsize=22)
axes[0].set_ylabel(r'$x_2$', fontsize=22)
axes[0].tick_params(axis='both', which='major', labelsize=21)


axes[1].set_title(r'Posterior of $x$', fontsize=22)
axes[1].hist2d(posterior_data[:, 0], posterior_data[:, 1], density=True, bins=100)
axes[1].set_xlabel(r'$x_1$',fontsize=22)
axes[1].set_ylabel(r'$x_2$', fontsize=22)
axes[1].tick_params(axis='both', which='major', labelsize=20)


fig.tight_layout()

# plt.show()
fig.savefig('posterior_theta_gammaprior.pdf')
