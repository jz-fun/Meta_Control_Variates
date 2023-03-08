
import torch
import higher
import time
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class NeuralCVModel(torch.nn.Module):
    def __init__(self, D_in,  h_dims, init_val):
        """
        :param D_in:
        :param h_dims:
        :param init_val:
        """
        super(NeuralCVModel, self).__init__()

        self.dims = h_dims # hidden dim
        self.dims.append(1) # output dim
        self.dims.insert(0, D_in) # input dim

        self.init_val = init_val

        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.dims[i-1], self.dims[i]) for i in range(1, len(self.dims))])

        self.c  = torch.nn.Parameter(init_val, requires_grad=True)


    def net_utils(self, x):
        x.requires_grad_(True)
        y = torch.sigmoid(self.layers[0](x))
        for it in range(1, len(self.layers) - 1):
            y = torch.sigmoid(self.layers[it](y))
        y = self.layers[-1](y)
        y = y * (1 - x).prod() * x.prod()
        grads = grad(y, x, create_graph=True)[0]
        x.requires_grad_(False)
        return y, grads



    def forward(self, x, score_x):
        """
        :param x:
        :param ind:
        :return:
        """
        # score = self.score_matrix[ind]
        score = score_x
        eva_net, grads = self.net_utils(x)

        y_pred =  self.c + grads.sum() + eva_net * score.sum()

        return y_pred


    def minibatch(self, x_batch, scores_x_batch):
        res = [self.forward(x, score_x) for x, score_x in zip(x_batch, scores_x_batch)]
        y_pred = torch.stack(res)
        return y_pred




class NeuralCV():
    def __init__(self,exp_name, model, D_in, h_dims, init_val, weight_decay, tasks, optim, lr, K, **kwargs_scorefunc):

        self.exp_name = exp_name
        self.tasks = tasks


        self.model = model
        self.net = self.model(D_in, h_dims, init_val)
        self.criterion = nn.MSELoss()


        score_func_and_params = []
        for kk, vv in kwargs_scorefunc.items():
            score_func_and_params.append(vv)
        self.score_function = score_func_and_params[0]
        self.score_func_parms = score_func_and_params[1:]

        # hyperparameters
        self.D_in = D_in
        self.h_dims = h_dims
        self.init_val = init_val
        self.weight_decay = weight_decay
        self.optim = optim
        self.lr =lr
        self.K = K

        # metrics
        self.plot_every = 10
        self.print_every = 10
        self.losses = []


    # iterables for training
    def chunks(self, ls, batch_size):
        """
        :param ls:
        :param batch_size:
        :return:
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]



    def train_val(self, num_epochs, batch_size, verbose = True):
        task = self.tasks.sample_task()
        true_val = task.true_integral_val()
        X, Y = task.sample_data(self.K)
        Score_X = self.score_function(self.score_func_parms[0], self.score_func_parms[1], X)
        X_qry, Y_qry = task.sample_data(self.K)
        Score_X_qry = self.score_function(self.score_func_parms[0], self.score_func_parms[1], X_qry)


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
                x, score_x = X[batch], Score_X[batch]
                y = Y[batch]
                opt.zero_grad()
                y_pred = self.net.minibatch(x, score_x)
                loss = (y - y_pred).pow(2).mean() + self.weight_decay * y_pred.pow(2).mean()
                loss.backward()
                opt.step()

            # After each epoch of training, compute the abserr of NCV on the training part
            self.net.eval()
            mcerr = (0.5*Y.detach().mean() + 0.5*Y_qry.detach().mean()- true_val).abs().item()

            # And Compute the abserr of NCV on the validation part
            if train_perc < 1.0:
                val_ncv_est = ( Y_qry.detach() - self.net.minibatch(X_qry,Score_X_qry).detach() +self.net.c.detach().squeeze()).mean().numpy()
                if verbose:
                    print('{}/{}th Epoch,  MC Abserr:{}, Validation AbserrNCV:{}'.format(i+1, num_epochs, \
                                                                                         mcerr, \
                                                                                         np.abs(val_ncv_est - true_val)))
            else:
                raise ValueError

            scheduler.step()
            np.random.shuffle(train_indices)

            # And Compute the estimates during training period
            if (i+1)%self.plot_every == 0:
                if train_perc < 1.0:
                    self.abserr_MC_est_log_overepoch.append(mcerr)
                    self.val_abserr_ncv_est_log_overepoch.append(np.abs(val_ncv_est- true_val))
                else:
                    raise ValueError
        return mcerr, np.abs(val_ncv_est- true_val)

