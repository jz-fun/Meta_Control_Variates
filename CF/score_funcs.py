
import torch
import numpy as np
import math
from matplotlib import pyplot as plt


def multivariate_uniform(dim, nullparm, X):

    n,d = X.size()[0], X.size()[1]
    assert dim == d, 'dim doesnot match'
    return torch.zeros(n,d)

def multivariate_Normal_score(mu, cov, X):
    """
    : calculate score value for all points in X, assuming they come from MVN(mu, cov)
    :param mu: d * 1
    :param cov: d * d
    :param X: 2d tensor, n * d, each row is a point
    :return: 2d tensor, n * d, each row is the grad_x logp(x)
    """
    assert mu.dim()  == 2 # as we always use torch.distributions.MultivariateNormal, note that this can also be used for a univariate normal
    assert cov.dim() == 2
    assert X.size()[1] == mu.size()[0] # should equal, the dim of a instance

    if X.size()[1] > 1:
        n = X.size()[0]
        d = X.size()[1]
        out = torch.zeros(n,d)
        for i in range(n):
            x_i = X[i].squeeze().unsqueeze(dim=1)
            assert x_i.size() == mu.size()
            out[i] = -1. * (torch.inverse(cov) @ (x_i - mu) ).squeeze()

    if X.size()[1] == 1:
        n = X.size()[0]
        d = X.size()[1]
        out = torch.zeros(n, d)
        for i in range(n):
            x_i = X[i].squeeze()
            out[i] = -1. * (torch.inverse(cov) @ (x_i - mu)).squeeze()

    return out
