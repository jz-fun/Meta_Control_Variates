
from CF.base_kernels import *
from CF.score_funcs import *
import torch




##########################################################################################
# 1. For single dataset: SGD on  log marginal likelihood
##########################################################################################
class negative_log_marginal_lik_MRI_singledat(torch.nn.Module):
    def __init__(self, prior_kernel, base_kernel, batch_size, X_whole_tr, Y_whole_tr, score_tensor_X_whole_tr, flag_if_use_medianheuristic):
        """
        Recall that we assume a zero mean function as prior mean function.
        Once v is optimized, in the SCV-optimization, we can pre-compute the k_0(X,X), for all datasets, i.e. k_0(X_1, X_1) ... k_0(X_1, X_T) ... k_0(X_T, X_T)
        ----
        :param prior_kernel: a class
        :param base_kernel:  a class
        :param X_whole_tr:   2d tensor, m * d
        :param Y_whole_tr:   2d tensor, m * 1
        :param score_tensor_X_whole_tr:  2d tensor, m * d
        :param flag_if_use_medianheuristic:
        """
        super(negative_log_marginal_lik_MRI_singledat, self).__init__()

        # Kernel hyper-parameters
        if flag_if_use_medianheuristic == True:
            if isinstance(base_kernel(), rbf_kernel) == True or isinstance(base_kernel(), rbf_boundcond_multd) == True:  # For rbf To avoid over-parametrization, set outputscale to 1
                self.base_kernel_parm1_raw = torch.log(torch.ones(1, dtype=torch.float32, requires_grad=False))
            else:
                self.base_kernel_parm1_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True)))

            self.base_kernel_parm2_raw = torch.log(torch.ones(1, dtype=torch.float32, requires_grad=False) *  torch.median(torch.cdist(X_whole_tr, X_whole_tr, p=2) ** 2) / torch.log(X_whole_tr.size()[0] * torch.ones(1))  )# Use median heuristic for the lengthscale


        if flag_if_use_medianheuristic == False:
            if isinstance(base_kernel(), rbf_kernel) == True or isinstance(base_kernel(), rbf_boundcond_multd) == True:  # For rbf To avoid over-parametrization, set outputscale to 1
                self.base_kernel_parm1_raw = torch.log(torch.ones(1, dtype=torch.float32, requires_grad=False))
            else:
                self.base_kernel_parm1_raw = torch.nn.Parameter( torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True)))  #

            self.base_kernel_parm2_raw = torch.nn.Parameter( torch.log( torch.ones(1, dtype=torch.float32, requires_grad=True) * 10 ))


        self.prior_kernel = prior_kernel # a class
        self.base_kernel = base_kernel   # a class
        self.X_whole_tr = X_whole_tr
        self.Y_whole_tr = Y_whole_tr
        self.score_tensor_X_whole_tr = score_tensor_X_whole_tr
        self.batch_size = batch_size



    def forward(self, batch_sample_indices, beta_cstkernel = 1):
        # assert len(batch_sample_indices) == self.batch_size

        self.base_kernel_parm1 = torch.exp(self.base_kernel_parm1_raw)
        self.base_kernel_parm2 = torch.exp(self.base_kernel_parm2_raw)

        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2

        X_batch = self.X_whole_tr[batch_sample_indices, :]
        Y_batch = self.Y_whole_tr[batch_sample_indices, :]
        score_batch = self.score_tensor_X_whole_tr[batch_sample_indices, :]

        k_XbXb = kernel_obj.cal_stein_base_kernel(X_batch, X_batch, score_batch, score_batch) + beta_cstkernel

        cond_number_threshold = 1e6
        lam = 1e-6
        bad_cond = np.linalg.cond(k_XbXb.detach().numpy()) >= cond_number_threshold
        k_Yb = k_XbXb + lam * torch.eye(X_batch.size()[0])
        while bad_cond:
            lam = 10 * lam
            k_Yb = k_XbXb + lam * torch.eye(X_batch.size()[0])
            bad_cond = np.linalg.cond(k_Yb.detach().numpy()) >= cond_number_threshold
        k_Yb.to(dtype=torch.float64)

        if Y_batch.dim() == 1:
            Y_batch = Y_batch.unsqueeze(dim=1)  # ensure Y is a column vector

        distrib = torch.distributions.MultivariateNormal(torch.zeros(Y_batch.size()[0]), covariance_matrix=k_Yb)
        log_mll = 0.
        for j in range(self.Y_whole_tr.size()[1]):
            log_mll += distrib.log_prob(Y_batch[:, j].squeeze())

        neg_log_mll = -1. * log_mll
        return neg_log_mll








class TuneKernelParams_mllk_MRI_singledat(object):
    def __init__(self, prior_kernel, base_kernel, X_train, Y_train, score_tensor):
        """
        :param prior_kernel: a class, stein
        :param base_kernel:  a class
        :param X_train:      2d tensor, m * d
        :param Y_train:      2d tensor, m * d
        :param score_tensor: 2d tensor, m * d
        """
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel

        self.X_train = X_train
        self.Y_train = Y_train
        self.score_tensor = score_tensor


    # split an iterable of items into batches
    def chunks(self, ls, batch_size):
        """
        Yield successive n-sized chunks from l.
        :params[in]: ls, an iterable of items
        :params[in]: batch_size, an integer, batch size
        returns a generator
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]  # note that the 'i+batch_size'th item is not included; thus we have 'batch_size' of samples



    def do_optimize_logmll(self, batch_size, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.01, epochs=100, verbose=True):
        neg_mll = negative_log_marginal_lik_MRI_singledat(self.prior_kernel, self.base_kernel, batch_size, self.X_train, self.Y_train, self.score_tensor, flag_if_use_medianheuristic)

        optimizer = torch.optim.Adam(neg_mll.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        m = self.X_train.size()[0]
        train_indices = list(range(m))

        for i in range(epochs):
            batches_generator = self.chunks(train_indices, batch_size)  # this creates a generator
            for batch_idx, batch_indices in enumerate(batches_generator):
                scheduler.step()
                optimizer.zero_grad()
                out = neg_mll(batch_indices, beta_cstkernel)
                out.backward()
                optimizer.step()


            # Random shuffle
            np.random.shuffle(train_indices)

            if verbose:
                print(i + 1, iter, out, neg_mll.base_kernel_parm1.detach(),neg_mll.base_kernel_parm2.detach())

        self.neg_mll = neg_mll


