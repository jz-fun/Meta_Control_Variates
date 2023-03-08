

import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import time




class rbf_kernel(object):
   # In the example, lengthscale l is a scalar.
   #       --- The kernel_parm2 is l^2 (to match the def. of median heuristic of l^2), the postivity of it are assured by exp transformation in the process of tuning them
    def __init__(self):
        self._kernel_parm1 = torch.ones(1) # convert to torch
        self._kernel_parm2 = torch.ones(1)

    @property
    def kernel_parm1(self):
        return self._kernel_parm1
    @kernel_parm1.setter
    def kernel_parm1(self, x):
        self._kernel_parm1 = x


    @property
    def kernel_parm2(self):
       return self._kernel_parm2
    @kernel_parm2.setter
    def kernel_parm2(self, x):
        self._kernel_parm2 = x


    def deriv_base_kernel(self, x, y):
        # return relevant derivatives of a pair of points including base kernel value at this pair
        # x: 1d tensor
        # y: 1d tensor
        ## difference between two tensors
        self.dim = x.size()[0] # ZS: as x is 1-d tensor here, so size[0] is its dim

        x_mins_y = x - y

        ## evaluation of kernel function at this pair of points k(x,y)
        ker_eval = self.kernel_parm1 * torch.exp(x_mins_y @ x_mins_y/(-2*self.kernel_parm2))


        ## partial derivative of kernel w.r.t. x
        ker_x = -1 * ker_eval * x_mins_y/self.kernel_parm2
        assert ker_x.size(0) == self.dim

        ## partial derivative of kernel w.r.t. y
        ker_y =  ker_eval * x_mins_y/self.kernel_parm2
        assert ker_y.size(0) == self.dim


        ## second derivative w.r.t. x and y // laplace
        ker_xy_vec = (1./self.kernel_parm2 - x_mins_y.pow(2)/self.kernel_parm2.pow(2)) * ker_eval
        assert ker_xy_vec.size(0) == self.dim
        ker_xy = ker_xy_vec.sum()

        ## return these items in the base kernel
        return (ker_eval, ker_x, ker_y, ker_xy)


    def cal_kernel(self, X1, X2):
        if len(X1.size()) == 1:   # as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
            X1 = X1.unsqueeze(1)  # suppose have m points
        if len(X2.size()) == 1:
            X2 = X2.unsqueeze(1)  # suppose have n points
        dist_mat = torch.cdist(X1, X2, p=2)**2    # m by n
        prior_covariance = self.kernel_parm1 * torch.exp(-0.5 * dist_mat / self.kernel_parm2)
        return prior_covariance













class rbf_boundcond_multd(object):

    def __init__(self):
        self._kernel_parm1 = 1
        self._kernel_parm2 = 1

    @property
    def kernel_parm1(self):
        return self._kernel_parm1

    @kernel_parm1.setter
    def kernel_parm1(self, x):
        self._kernel_parm1 = x

    @property
    def kernel_parm2(self):
        return self._kernel_parm2

    @kernel_parm2.setter
    def kernel_parm2(self, x):
        self._kernel_parm2 = x



    def deriv_base_kernel_noboundcond(self, x, y):
        # return relevant derivatives of a pair of points including base kernel value at this pair
        # x: 1d tensor
        # y: 1d tensor
        self.dim = x.size()[0] # ZS: as x is 1-d tensor here, so size[0] is its dim

        x_mins_y = x - y

        ## evaluation of kernel function at this pair of points k(x,y)
        ker_eval = self.kernel_parm1 * torch.exp(x_mins_y @ x_mins_y/(-2*self.kernel_parm2))


        ## partial derivative of kernel w.r.t. x
        ker_x = -1 * ker_eval * x_mins_y/self.kernel_parm2
        assert ker_x.size(0) == self.dim

        ## partial derivative of kernel w.r.t. y
        ker_y =  ker_eval * x_mins_y/self.kernel_parm2
        assert ker_y.size(0) == self.dim


        ## second derivative w.r.t. x and y // laplace
        ker_xy_vec = (1./self.kernel_parm2 - x_mins_y.pow(2)/self.kernel_parm2.pow(2)) * ker_eval
        assert ker_xy_vec.size(0) == self.dim
        ker_xy = ker_xy_vec.sum()

        ## return these items in the base kernel
        return (ker_eval, ker_x, ker_y, ker_xy)



    def helper_func_bound(self, x, y):
        ## a is a 1d tensor; default is zero-valued as this is tailored for windfarm example
        delta_x = (x*(1-x)).prod()
        delta_y = (y*(1-y)).prod()

        # i. boundary_func
        delta_x_delta_y = delta_x * delta_y

        # ii. 1st order deriv
        nabla_x_deltaxdeltay = torch.zeros(self.dim)
        nabla_y_deltaxdeltay = torch.zeros(self.dim)
        for j in range(self.dim):
            nabla_x_deltaxdeltay[j] = (1-2*x[j])*delta_x_delta_y/(x[j] * (1-x[j]))
            nabla_y_deltaxdeltay[j] = (1-2*y[j])*delta_x_delta_y/(y[j] * (1-y[j]))

        # iii.
        nablay_nablax_deltaxdeltay = torch.zeros(1)
        for j in range(self.dim):
            temp_de = (x[j] * (1-x[j]))*(y[j] * (1-y[j]))
            temp_ne = (1-2*x[j])*(1-2*y[j])
            nablay_nablax_deltaxdeltay += temp_ne * delta_x_delta_y/temp_de

        # all five tensors are 1d tensor
        return delta_x_delta_y, nabla_x_deltaxdeltay, nabla_y_deltaxdeltay, nablay_nablax_deltaxdeltay


    def deriv_base_kernel(self, x, y):
        # Add here boundary condition
        #     --- by using self.helper_func_bound and self.deriv_base_kernel_noboundcond
        ker_eval, ker_x, ker_y, ker_xy = self.deriv_base_kernel_noboundcond(x,y)
        delta_x_delta_y, nabla_x_deltaxdeltay, nabla_y_deltaxdeltay, nablay_nablax_deltaxdeltay = self.helper_func_bound(x, y)

        modi_ker_eval = ker_eval * delta_x_delta_y
        modi_ker_x = delta_x_delta_y * ker_x + ker_eval * nabla_x_deltaxdeltay
        modi_ker_y = delta_x_delta_y * ker_y + ker_eval * nabla_y_deltaxdeltay
        modi_ker_xy = ker_xy * delta_x_delta_y + (ker_x * nabla_y_deltaxdeltay).sum() + \
                      (ker_y * nabla_x_deltaxdeltay).sum() + ker_eval * nablay_nablax_deltaxdeltay

        return (modi_ker_eval, modi_ker_x, modi_ker_y, modi_ker_xy)



    def cal_kernel(self, X1, X2):
        if len(X1.size()) == 1:  # as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
            X1 = X1.unsqueeze(1)  # suppose have m points
        if len(X2.size()) == 1:
            X2 = X2.unsqueeze(1)  # suppose have n points
        dist_mat = torch.cdist(X1, X2, p=2) ** 2  # m by n
        prior_covariance = self.kernel_parm1 * torch.exp(-0.5 * dist_mat / self.kernel_parm2)
        return prior_covariance

