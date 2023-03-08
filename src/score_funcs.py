
import torch
import numpy as np
import math
from matplotlib import pyplot as plt



def multivariate_Normal_score(mu, cov, X):
    """
    calculate score value for all points in X, assuming they come from MVN(mu, cov)
    :param mu: d * 1
    :param cov: d * d
    :param X: 2d tensor, n * d, each row is a point
    :return: 2d tensor, n * d, each row is the grad_x logp(x)
    """
    assert mu.dim()  == 2 # as we always use torch.distributions.MultivariateNormal, note that this can also be used for a univariate normal
    assert cov.dim() == 2
    assert X.size()[1] == mu.size()[0] # should equal, the dim of a instance

    n = X.size()[0]
    d = X.size()[1]
    out = torch.zeros(n, d)

    if X.size()[1] > 1:
        for i in range(n):
            x_i = X[i].squeeze().unsqueeze(dim=1)
            assert x_i.size() == mu.size()
            out[i] = -1. * (torch.inverse(cov) @ (x_i - mu) ).squeeze()

    if X.size()[1] == 1:
        for i in range(n):
            x_i = X[i].squeeze()
            out[i] = -1. * (torch.inverse(cov) @ (x_i - mu)).squeeze()

    return out




def multivariate_Uniform_score(dim, null_param, X):
    """
    :param dim: dim of x
    :param X:   design matrix, each of which row is x_i
    :return:    scores of uniform: 2d tensor of size n by dim
    """
    assert dim == X.size()[1] , "Dim of samples doesnot match. Check dim and your design matrix X."
    n  = X.size()[0]
    return torch.zeros(n, dim)




def univariate_log_Normal_score(mu, sigma, X):
    n = X.size()[0]
    out = torch.zeros(n,1)
    for i in range(0,n,1):
        out[i] = -1/X[i] - (torch.log(X[i]) - mu)/(X[i]*(sigma**2))
    return out


def multivariate_log_Normal_score(mu, S, X):
    """

    :param mu: 2d tensor of size d * 1
    :param S: 2d tensor of size d * d; which is a diagonal matrix; each element on the diagonal is a sigma
    :param X: 2d tensor of size n * d
    :return:
    """
    assert X.size()[1] == mu.size()[0] and X.size()[1] == S.size()[0], "wrong dim"
    assert S.size()[0] == S.size()[1], 'S should be a square matrix'
    n = X.size()[0]
    d = X.size()[1]
    out = torch.zeros(n, d)
    for i in range(0,n,1):
        for j in range(0, d, 1):
            out[i,j] = -1/X[i, j] - (torch.log(X[i, j]) - mu[j])/(X[i,j]*(S[j,j]**2))
    return out

