
import torch
import time
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
import random


class NeuralCVModel_lotka(torch.nn.Module):
    def __init__(self, D_in,  h_dims, init_val):
        """
        :param D_in:
        :param h_dims:
        :param init_val:
        """
        super(NeuralCVModel_lotka, self).__init__()

        self.dims = h_dims # hidden dim
        self.dims.append(1) # output dim
        self.dims.insert(0, D_in) # input dim

        self.dims.insert(len(h_dims), 8)

        self.init_val = init_val
        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.dims[i-1], self.dims[i]) for i in range(1, len(self.dims))])

        self.c  = torch.nn.Parameter(init_val, requires_grad=True)


    def singleforwardpass(self,x):
        # use with u:R^d-> R^d
        y = torch.tanh(self.layers[0](x))
        for it in range(1, len(self.layers) - 1):
            y = torch.tanh(self.layers[it](y))
        y = self.layers[-1](y)
        return y


    def net_utils(self, x):
        #  use with u:R^d-> R^d
        grads = torch.autograd.functional.jacobian(self.singleforwardpass, x).detach().diag()
        y = self.singleforwardpass(x)

        return y, grads


    def forward(self, x, score_x):
        """
        :param x:
        :param ind:
        :return:
        """
        score = score_x
        eva_net, grads = self.net_utils(x)

        #  use with u:R^d-> R^d
        y_pred = self.c + grads.sum() + torch.dot(eva_net, score)
        return y_pred


    def minibatch(self, x_batch, scores_x_batch):
        res = [self.forward(x, score_x) for x, score_x in zip(x_batch, scores_x_batch)]
        y_pred = torch.stack(res)
        return y_pred




class NeuralCV_lotka():
    def __init__(self,parma_idx, tasks, model, D_in, h_dims, weight_decay, optim, lr, K=10):
        self.tasks = tasks
        self.parma_idx = parma_idx

        self.model = model
        self.net = self.model(D_in, h_dims, torch.zeros(1))

        self.criterion = nn.MSELoss()

        # hyperparameters
        self.D_in = D_in
        self.h_dims = h_dims
        self.optim = optim
        self.lr = lr

        self.K = K
        self.weight_decay = weight_decay

        # metrics
        self.plot_every = 10
        self.print_every = 10
        self.meta_losses = []


    def helper_get_parm(self):
        # use to choose the parameter of interest to infer its posterior mean
        return self.parma_idx

    # iterables for training
    def chunks(self, ls, batch_size):
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]



    def train_val(self,task_id,num_epochs, batch_size, norm_init_std=1e-3, verbose = True):
       # task_id: help to identify task number in the giant meta dataset "self.tasks"
        parm_idx = self.helper_get_parm()

        true_val = self.tasks['X_all'][task_id][:,parm_idx].mean().detach()

        li = list(range(0, self.tasks['X_all'][task_id].size()[0]))
        support_idx = random.sample(li, self.K)
        X = self.tasks['X_all'][task_id][support_idx].clone().detach()
        Y = self.tasks['X_all'][task_id][support_idx, parm_idx].clone().detach()
        score_X = self.tasks['score_unconstrainedsamples'][task_id][support_idx].clone().detach()

        for each_par in self.net.parameters():  ## initialize this neural nets
            if each_par is self.net.c:
                torch.nn.init.constant_(each_par, Y.mean().detach())
            torch.nn.init.normal_(each_par, mean=0, std=norm_init_std)

        li_q = list(set(li) - set(support_idx))
        query_idx = random.sample(li_q, self.K)
        X_qry = self.tasks['X_all'][task_id][query_idx].clone().detach()
        Y_qry = self.tasks['X_all'][task_id][query_idx, parm_idx].clone().detach()
        score_X_qry = self.tasks['score_unconstrainedsamples'][task_id][query_idx].clone().detach()


        opt = self.optim(self.net.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)

        # Used for log
        self.train_abserr_ncv_log_overepoch = []
        self.val_abserr_ncv_est_log_overepoch = []
        self.val_indicator_log_overepoch = []
        self.abserr_MC_est_log_overepoch = []

        n_train, n_val = X.size()[0], X_qry.size()[0]
        train_indices = np.arange(n_train)
        train_perc = n_train/(n_train + n_val)


        for i in range(num_epochs):
            batches = self.chunks(train_indices, batch_size)
            self.net.train(True)
            for batch in batches:
                x, score_x = X[batch], score_X[batch]
                y = Y[batch]
                opt.zero_grad()
                y_pred = self.net.minibatch(x, score_x)
                loss = (y - y_pred).pow(2).mean() + self.weight_decay * y_pred.pow(2).mean()
                loss.backward()
                opt.step()


            scheduler.step()
            np.random.shuffle(train_indices)

        self.net.eval()
        mcerr = (0.5*Y.detach().mean() + 0.5*Y_qry.detach().mean()- true_val).abs().item()
        val_ncv_est = (Y_qry.detach() - self.net.minibatch(X_qry,score_X_qry).detach()).mean()

        return mcerr, (val_ncv_est- true_val).abs()


