import torch
# Kernel
def base_kernel(X1, X2, kerparms):
    if len(X1.size()) == 1:  #: as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
        X1 = X1.unsqueeze(1)
    if len(X2.size()) == 1:
        X2 = X2.unsqueeze(1)

    kernel_parm1 = kerparms[0]
    kernel_parm2 = kerparms[1]

    dist_mat = torch.cdist(X1, X2, p=2) ** 2

    m = X1.size()[0]
    n = X2.size()[0]


    norms_X1 = X1.norm(dim=1, p=2).pow(2)  # as we assume each row represents a point, we compute norm by rows.
    norms_X2 = X2.norm(dim=1, p=2).pow(2)

    norms_X1 = norms_X1.unsqueeze(dim=1)  # size is [m,1]
    norms_X2 = norms_X2.unsqueeze(dim=0)  # size is [1,n]

    mat = (1 + kernel_parm1 * norms_X1.repeat(1, n)) * (1 + kernel_parm1 * norms_X2.repeat(m, 1))

    prior_covariance = (1 / (mat)) * torch.exp(-0.5 * dist_mat / kernel_parm2 ** 2)
    return prior_covariance


def rbf_kernel(X1, X2, kerparms):
    if len(X1.size()) == 1:  # as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
        X1 = X1.unsqueeze(1)  # suppose have m points
    if len(X2.size()) == 1:
        X2 = X2.unsqueeze(1)  # suppose have n points

    kernel_parm1 = kerparms[0]
    kernel_parm2 = kerparms[1]

    dist_mat = torch.cdist(X1, X2, p=2) ** 2  # m by n


    prior_covariance = kernel_parm1 * torch.exp(-0.5 * dist_mat / kernel_parm2)
    return prior_covariance



