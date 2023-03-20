
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


    # Tune kernel hyper-parameters w.r.t. log marginal likelihood
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


