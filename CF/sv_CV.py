
import torch
import numpy as np
from CF.score_funcs import *
from CF.base_kernels import *
from CF.stein_operators import *
from CF.utils import *

class Simplied_CF(object):

    def __init__(self, prior_kernel, base_kernel, X_train, Y_train, score_tensor):
        """
        :param prior_kernel: a kernel class, here is a stein kernel class.
        :param base_kernel: a kernel class
        :param X_train:  2d tensor, m * d
        :param Y_train:  2d tensor, m *  1
        :param score_tensor:  2d tensor, m * d
        """
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.X_train = X_train
        self.Y_train = Y_train
        self.score_tensor = score_tensor


    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_singledat(self.prior_kernel, self.base_kernel,  self.X_train, self.Y_train, self.score_tensor)
        tune_kernelparams_negmllk_obj.do_optimize_logmll(batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr, epochs, verbose)
        optim_base_kernel_parms = torch.Tensor([tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm1, tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm2])
        self.optim_base_kernel_parms =optim_base_kernel_parms.detach()
        return optim_base_kernel_parms.detach()


    def do_closed_form_est_for_simpliedCF(self):
        # Simplified CF estimate
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
        k_XX = kernel_obj.cal_stein_base_kernel(self.X_train, self.X_train, self.score_tensor, self.score_tensor)
        m = self.X_train.size()[0]
        o  = (torch.ones(1, m )  @ (k_XX + 0.001 * torch.eye(m)).inverse() @ self.Y_train )/( torch.ones(1, m)  @ (k_XX + 0.001 * torch.eye(m)).inverse() @ torch.ones( self.X_train.size()[0], 1 )  )
        return o


    def do_nonsim_CF(self, X_te, Y_te, score_tensor):
        # Simplified CF estimate
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
        k_ZX =  kernel_obj.cal_stein_base_kernel(X_te, self.X_train, score_tensor, self.score_tensor)
        k_XX = kernel_obj.cal_stein_base_kernel(self.X_train, self.X_train, self.score_tensor, self.score_tensor)
        n = X_te.size()[0]
        m = self.X_train.size()[0]
        o  = (torch.ones(1, m )  @ (k_XX + 0.001 * torch.eye(m)).inverse() @ self.Y_train )/( torch.ones(1, m)  @ (k_XX + 0.001 * torch.eye(m)).inverse() @ torch.ones( self.X_train.size()[0], 1 )  )
        fit = k_ZX @  (k_XX + 0.001 * torch.eye(m)).inverse() @ (self.Y_train.squeeze()-o).squeeze()
        I=  (Y_te.squeeze() - fit.squeeze()).mean()
        return I






# # Experiments
# dim = 1
# factor = torch.ones(1) * 1
# mu = torch.zeros(dim, dtype=torch.float) + 0
# var = torch.eye(dim, dtype=torch.float) * factor
# def my_func_1(X):
#     return (0.5 + (2 * (X >= 0) - 1) * 1.5) * torch.ones(1, dtype=torch.float)
# X1 = mu + torch.sqrt(factor) * torch.randn(50, dim)
# Y1 = my_func_1(X1)
# mu = torch.zeros(dim, 1)
# cov = var
# score_X1 = multivariate_Normal_score(mu, cov, X1)
#
# myCF = Simplied_CF(stein_base_kernel_MV_2, rbf_kernel, X1, Y1, score_X1)
# myCF.do_tune_kernelparams_negmllk(batch_size_tune=10, flag_if_use_medianheuristic=False, beta_cstkernel=0, lr=0.1, epochs=20, verbose=True)
# simp_CF_est = myCF.do_closed_form_est_for_simpliedCF()
# MC_est = Y1.mean()
#
#
#
# X2 = mu + torch.sqrt(factor) * torch.randn(50, dim)
# Y2 = my_func_1(X2)
# score_X2 = multivariate_Normal_score(mu, cov, X2)
# I = myCF.do_nonsim_CF(X2,Y2, score_X2)


# def my_func_1(X):
#     return X.sum(1).squeeze().unsqueeze(1)
# a = torch.tensor([0.125, 5, 100])
# b = torch.tensor([0.5, 15, 200])
# X1 = (b-a)*torch.rand(10,3)+a
# score_X1 = multivariate_uniform(3, None, X1)
# Y1 = my_func_1(X1)
# myCF = Simplied_CF(stein_base_kernel_MV_2, rbf_boundcond_tsu, X1, Y1, score_X1)
# myCF.do_tune_kernelparams_negmllk(batch_size_tune=10, flag_if_use_medianheuristic=False, beta_cstkernel=0, lr=0.1, epochs=20, verbose=True)
# simp_CF_est = myCF.do_closed_form_est_for_simpliedCF()
# MC_est = Y1.mean()
# print(simp_CF_est, MC_est)
#
#
# # 重复实验
# out = torch.zeros(20,2)
# for i in range(20):
#     print('{}/{}'.format(i+1, 20))
#     # a = torch.tensor([0.125, 5, 100])
#     # b = torch.tensor([0.5, 15, 200])
#     a = torch.tensor([0., 0., 0.])
#     b = torch.tensor([1., 1., 1.])
#     X1 = (b - a) * torch.rand(10, 3) + a
#     score_X1 = multivariate_uniform(3, None, X1)
#     Y1 = my_func_1(X1)
#     myCF = Simplied_CF(stein_base_kernel_MV_2, rbf_boundcond_tsu, X1, Y1, score_X1)
#     myCF.do_tune_kernelparams_negmllk(batch_size_tune=10, flag_if_use_medianheuristic=False, beta_cstkernel=0, lr=0.1,
#                                       epochs=20, verbose=True)
#     # myCF = Simplied_CF(stein_base_kernel_MV_2, rbf_boundcond_tsu, X1, Y1, score_X1)
#     # myCF.do_tune_kernelparams_negmllk(batch_size_tune=10, flag_if_use_medianheuristic=False, beta_cstkernel=0, lr=0.1,
#     #                                   epochs=20, verbose=False)
#     out[i,0] = myCF.do_closed_form_est_for_simpliedCF()
#     out[i,1] = Y1.mean()
#
# # Mean absolute error
# (out - 1.5).abs().mean(0)