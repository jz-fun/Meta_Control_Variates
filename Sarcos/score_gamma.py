import torch

# Note the differences in definition of Gamma distribution in Pytorch and Matlab
# Define score_function
def multivariate_Gamma_score(alphaS, betaS, X):
    """
    :param alphaS: 1d tensor of size d
    :param betaS: 1d tensor of size d
    :param X:     2d tensor of size n ,d
    :return:
    """
    n  = X.size()[0]
    d =  X.size()[1]
    out = torch.zeros(n,d)
    for i in range(n):
        out[i] = (alphaS-1)/X[i] - betaS
    return out.float()


