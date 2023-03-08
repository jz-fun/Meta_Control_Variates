

import higher

from src.score_funcs import *

import time
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np



class MetaNeuralCVModel(torch.nn.Module):
    def __init__(self, D_in,  h_dims, init_val):
        """
        :param D_in:
        :param h_dims:
        :param score_matrix:
        :param init_val:
        :param K:
        """
        super(MetaNeuralCVModel, self).__init__()

        self.dims = h_dims # hidden dim
        self.dims.append(1) # output dim
        self.dims.insert(0, D_in) # input dim


        self.init_val = init_val


        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.dims[i-1], self.dims[i]) for \
                                      i in range(1, len(self.dims))])

        self.c = torch.nn.Parameter(init_val, requires_grad=True)




    def net_utils(self, x):

        x.requires_grad_(True)
        y = torch.sigmoid(self.layers[0](x))
        for it in range(1, len(self.layers) - 1):
            y = torch.sigmoid(self.layers[it](y))
        y = self.layers[-1](y)  # last layer -- linear


        y = y * (1-x).prod() * x.prod()
        grads = grad(y, x, create_graph=True)[0]
        x.requires_grad_(False)
        return y, grads


    def forward(self, x, score_x):
        """
        :param x:
        :param ind:
        :return:
        """

        score = score_x
        eva_net, grads = self.net_utils(x)
        y_pred = self.c + grads.sum() + eva_net * score.sum()

        return y_pred


    def minibatch(self, x_batch, scores_x_batch):
        res = [self.forward(x, score_x) for x, score_x in zip(x_batch, scores_x_batch)]
        y_pred = torch.stack(res)
        return y_pred


    def chunks(self, ls, batch_size):
        """
        :param ls:
        :param batch_size:
        :return:
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:1 + batch_size]


class MetaNeuralCV():
    def __init__(self, exp_name, model, D_in, h_dims, init_val, weight_decay, tasks, inner_optim, inner_lr, \
                 meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=100, **kwargs_scorefunc):

        # important objects
        self.exp_name = exp_name
        self.tasks = tasks

        self.model = model
        self.net = self.model(D_in, h_dims, init_val)

        self.criterion = nn.MSELoss()

        # From the dictionary '' kwargs_scorefunc '' to get the distrbutional parameters required for 'score_function'
        #    If it is a Gaussian, the first element would be mean; the second would be cov
        score_func_and_params = []
        for kk, vv in kwargs_scorefunc.items():
            score_func_and_params.append(vv)
        self.score_function = score_func_and_params[0]
        self.score_func_parms = score_func_and_params[1:]


        # hyperparameters
        self.D_in = D_in
        self.h_dims = h_dims
        self.init_val = init_val

        self.inner_optim = inner_optim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch
        self.weight_decay = weight_decay

        # metrics
        self.plot_every = 10
        self.print_every = 10  
        self.meta_losses = []


    def main_loop(self, num_iterations):
        self.net.train()

        inner_opt = self.inner_optim(self.net.parameters(), lr=self.inner_lr)
        meta_opt = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        epoch_loss = 0
        pointer1 = 0

        for iteration in range(1, num_iterations + 1):
            meta_loss = 0
            meta_opt.zero_grad()

            pointer1 += 1

            for i in range(self.tasks_per_meta_batch):
                if self.exp_name == 'oscillatory':
                    task = self.tasks.sample_task()
                    X, y = task.sample_data(self.K)
                    score_X = self.score_function(self.score_func_parms[0], self.score_func_parms[1], X)

                with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                    # NB: 'copy_initial_weights=False' is default setting for MAML
                    for _ in range(self.inner_steps):
                        _y_pred = fnet.minibatch(X, score_X).squeeze()
                        # print(_y_pred)
                        loss = self.criterion(_y_pred, y.squeeze()) / self.K + self.weight_decay * _y_pred.pow(2).mean()
                        diffopt.step(loss)

                    # Get query set
                    if self.exp_name == 'oscillatory':
                        X_qry, y_qry = task.sample_data(self.K)
                        score_X_qry = self.score_function(self.score_func_parms[0], self.score_func_parms[1], X_qry)

                    _y_pred_qry = fnet.minibatch(X_qry, score_X_qry).squeeze()
                    # print(_y_pred_qry)
                    val_loss = self.criterion(_y_pred_qry, y_qry.squeeze()) / self.K + self.weight_decay * _y_pred_qry.pow(2).mean()
                    val_loss.backward()
                    meta_loss += val_loss.detach() / self.tasks_per_meta_batch  # ZS: get the average here


            meta_opt.step()
            # log metrics
            epoch_loss += meta_loss.item()

            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0



    def test(self, n_total_tasks, inner_optim, inner_steps=3):
        self.log_test = []

        self.net.train() # in higher

        self.qry_losses = []
        self.meta_cv_ests = []
        self.abs_err_meta_cv_ests = []

        self.meta_cv_simp_ests = []
        self.abs_err_meta_cv_simp_ests = []

        self.true_vals = []

        self.MC_ests = []
        self.abs_err_MC_ests = []

        self.MC_2m_ests = []
        self.abs_err_MC_2m_ests = []

        inner_opt = inner_optim(self.net.parameters(), lr=self.inner_lr)

        for i in range(n_total_tasks):
            task = self.tasks.sample_task()
            true_val = task.true_integral_val()
            self.true_vals.append(true_val)
            X, y = task.sample_data(self.K)

            score_X = self.score_function(self.score_func_parms[0], self.score_func_parms[1], X)
            meta_loss = 0
            with higher.innerloop_ctx(self.net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                for _ in range(inner_steps):
                    _y_pred = fnet.minibatch(X, score_X).squeeze()
                    loss = self.criterion(_y_pred, y.squeeze()) / self.K + self.weight_decay * _y_pred.pow(2).mean()
                    diffopt.step(loss)

                # generate query set
                X_qry, y_qry = task.sample_data(self.K)
                score_X_qry = self.score_function(self.score_func_parms[0], self.score_func_parms[1], X_qry)
                _y_pred_qry = fnet.minibatch(X_qry, score_X_qry).squeeze()
                val_loss = self.criterion(_y_pred_qry, y_qry.squeeze()) / self.K + self.weight_decay * _y_pred_qry.pow(2).mean()
                # val_loss = self.criterion(_y_pred_qry, y_qry.squeeze()) / self.K

                meta_loss += val_loss / self.tasks_per_meta_batch  # ZS: get the average here

                # log numbers
                self.qry_losses.append(meta_loss.detach())

                self.MC_ests.append(y_qry.squeeze().mean().detach().numpy())
                self.MC_2m_ests.append(0.5 * (y_qry.squeeze().mean().detach().numpy() + y.squeeze().mean().detach().numpy()))
                self.meta_cv_ests.append((y_qry.squeeze() - _y_pred_qry.detach() + self.net.c.detach().squeeze() ).mean().detach().numpy())
                

                self.abs_err_MC_ests.append(np.abs(self.MC_ests[i] - self.true_vals[i]))
                self.abs_err_MC_2m_ests.append(np.abs(self.MC_2m_ests[i] - self.true_vals[i]))
                self.abs_err_meta_cv_ests.append(np.abs(self.meta_cv_ests[i] - self.true_vals[i]))
                

        
        print('Meta_testing finished.')
        self.log_test.append({
            'n_total_tasks': n_total_tasks,
            'loss': self.qry_losses,
            'MC_2m_ests': self.MC_2m_ests,
            'Abserr_CVests': self.abs_err_meta_cv_ests,
            'Abserr_MCests': self.abs_err_MC_ests,
            'Abserr_MC_2m_ests': self.abs_err_MC_2m_ests,
            'mode': 'test',
            'time': time.time(),
        })

