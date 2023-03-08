
import torch
import higher
import time
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import random



class MetaNeuralCVModel_lotka(torch.nn.Module):
    def __init__(self, D_in,  h_dims, init_val, K):
        """
        :param D_in:
        :param h_dims:
        :param init_val:
        :param K:
        """
        super(MetaNeuralCVModel_lotka, self).__init__()
        self.K = K

        self.dims = h_dims # hidden dim
        self.dims.append(1) # output dim
        self.dims.insert(0, D_in) # input dim


        # use with u:R^d-> R^d
        self.dims.insert(len(h_dims), 8)

        # self.score_matrix = score_matrix
        self.init_val = init_val

        self.layers = torch.nn.ModuleList([torch.nn.Linear(self.dims[i-1], self.dims[i]) for \
                                      i in range(1, len(self.dims))])

        self.c  = torch.nn.Parameter(init_val, requires_grad=True)


    def singleforwardpass(self,x):
        # use with u:R^d-> R^d
        y = torch.tanh(self.layers[0](x))
        for it in range(1, len(self.layers) - 1):
            y = torch.tanh(self.layers[it](y))
        y = self.layers[-1](y)
        return y


    def net_utils(self, x):
        # use with u:R^d-> R^d
        grads = torch.autograd.functional.jacobian(self.singleforwardpass, x).detach().diag()
        y = self.singleforwardpass(x)

        return y, grads



    def forward(self, x, score_x):
        score = score_x
        eva_net, grads = self.net_utils(x)
        # use with u:R^d-> R^d
        y_pred = self.c + grads.sum() + torch.dot(eva_net, score)

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


class MetaNeuralCV_lotka():
    def __init__(self, parma_idx, model, D_in, h_dims, init_val, weight_decay, tasks, inner_optim, inner_lr, \
                 meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=100):

        # important objects
        self.tasks = tasks
        self.parma_idx =parma_idx

        self.model = model
        self.net = self.model(D_in, h_dims, init_val, K)

        self.criterion = nn.MSELoss()

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



    def helper_get_parm(self):
        # use to choose the parameter of interest to infer its posterior mean
        return self.parma_idx

    def train_val_case2(self, num_iterations, lotka_ifalluse_allobs, ts, te, meta_val_datasets=None, norm_init_std=1e-3, verbose=True):

        parm_idx = self.helper_get_parm()
        for each_par in self.net.parameters():   ## initialize this neural nets
            if each_par is self.net.c:
                pass
                # torch.nn.init.constant_(each_par, somevalue)
            torch.nn.init.normal_(each_par, mean=0, std=norm_init_std)

        # Used for log
        self.meta_val_indicator_log_overepoch = []
        self.meta_val_loss_log_overepoch = []
        self.meta_val_abserr_mcv_est_log_overepoch = []
        self.meta_val_abserr_mcv_sim_est_log_overepoch = []
        self.meta_val_abserr_MC_est_log_overepoch = []
        self.meta_val_abserr_MC_2m_est_log_overepoch = []

        # Use for log
        n_rep = num_iterations // self.plot_every
        if meta_val_datasets is not None:
            meta_val_indicator_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_loss_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_truevalues_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_mcv_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_abserr_mcv_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_mcv_sim_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_abserr_mcv_sim_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_MC_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))  # num_iterations
            meta_val_MC_2m_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_abserr_MC_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))
            meta_val_abserr_MC_2m_est_log = torch.zeros(n_rep, len(meta_val_datasets['X']))


        self.net.train()
        inner_opt = self.inner_optim(self.net.parameters(), lr=self.inner_lr)
        meta_opt = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(meta_opt, step_size=10, gamma=0.9)

        epoch_loss = 0
        pointer1 = 0
        for iteration in range(1, num_iterations + 1):
            meta_loss = 0
            meta_opt.zero_grad()

            pointer1 += 1
            for i in range(self.tasks_per_meta_batch):
                if (self.D_in == 8) and (lotka_ifalluse_allobs ==False):
                    # when D_in == 8, means this is Lotka example; and now self.tasks is a dataset rather than a task generator (prestored)
                    # shuffle the data once all store tasks have been passed through
                    #     AND reset the pointer1
                    if (pointer1 != 1) and (pointer1 % (len(self.tasks['X']) // self.tasks_per_meta_batch) == 1):
                        #  to make this reproducible , fix the random seed for shuffling
                        np.random.seed(iteration)
                        indices_shuffle = np.arange(len(self.tasks['X']))
                        np.random.shuffle(indices_shuffle)
                        self.tasks['X'] = [self.tasks['X'][i] for i in indices_shuffle]
                        self.tasks['FX'] = [self.tasks['FX'][i] for i in indices_shuffle]
                        self.tasks['score_X'] = [self.tasks['score_X'][i] for i in indices_shuffle]
                        pointer1 = 1

                li = range(0, self.tasks['X_all'][i].size()[0] )
                support_idx = random.sample(li,self.K)
                X = self.tasks['X_all'][(pointer1 - 1) * self.tasks_per_meta_batch + i][support_idx].clone().detach()
                y = self.tasks['X_all'][(pointer1 - 1) * self.tasks_per_meta_batch + i][support_idx, parm_idx].clone().detach()
                score_X = self.tasks['score_unconstrainedsamples'][(pointer1 - 1) * self.tasks_per_meta_batch + i][support_idx].clone().detach()


                for each_par in self.net.parameters():
                    if each_par is self.net.c:
                        torch.nn.init.constant_(each_par, y.detach().mean())


                with higher.innerloop_ctx(self.net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                    # 'copy_initial_weights=False' is default setting for MAML
                    # generate query set
                    if self.D_in == 8:
                        query_idx = random.sample(li, self.K)
                        X_qry = self.tasks['X_all'][(pointer1 - 1) * self.tasks_per_meta_batch + i][query_idx].clone().detach()
                        y_qry = self.tasks['X_all'][(pointer1 - 1) * self.tasks_per_meta_batch + i][query_idx,parm_idx].clone().detach()
                        score_X_qry = self.tasks['score_unconstrainedsamples'][(pointer1 - 1) * self.tasks_per_meta_batch + i][query_idx].clone().detach()


                    for _ in range(self.inner_steps):
                        _y_pred = fnet.minibatch(X, score_X).squeeze()
                        loss = self.criterion(_y_pred, y.squeeze()) / self.K + self.weight_decay * _y_pred.pow(2).mean()
                        diffopt.step(loss)

                    _y_pred_qry = fnet.minibatch(X_qry, score_X_qry).squeeze()
                    val_loss = self.criterion(_y_pred_qry, y_qry.squeeze()) / self.K + self.weight_decay * _y_pred_qry.pow(2).mean()
                    val_loss.backward()
                    meta_loss += val_loss.detach() / self.tasks_per_meta_batch  #get the average here


            meta_opt.step()

            # log metrics
            epoch_loss += meta_loss.detach().item()
            scheduler.step()
            if meta_val_datasets is None and verbose == True and iteration % self.plot_every == 0:
                print(f'{iteration}/{num_iterations} | Meta_Train_Loss: {epoch_loss / self.plot_every:.4f}')


            if iteration % self.plot_every == 0 and meta_val_datasets is not None:
                self.net.train()
                r = (iteration//self.plot_every)-1


                meta_val_loss = 0
                for kk in range(len(meta_val_datasets['X'])):
                    li = list(range(0, self.tasks['X_all'][kk].size()[0]))
                    support_idx = random.sample(li, self.K)
                    #
                    X = self.tasks['X_all'][kk][support_idx].clone().detach()
                    y = self.tasks['X_all'][kk][support_idx, parm_idx].clone().detach()
                    score_X = self.tasks['score_unconstrainedsamples'][kk][support_idx].clone().detach()

                    for each_par in self.net.parameters():  ## initialize this neural nets
                        if each_par is self.net.c:
                            torch.nn.init.constant_(each_par, y.detach().mean())


                    with higher.innerloop_ctx(self.net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                        # track_higher_grads=False used in test mode
                        for _ in range(self.inner_steps):
                            _y_pred = fnet.minibatch(X, score_X).squeeze()
                            loss = self.criterion(_y_pred, y.squeeze()) / self.K + self.weight_decay * _y_pred.pow(2).mean()

                            diffopt.step(loss)
                        # Get query set
                        li_q = list(set(li)-set(support_idx))
                        query_idx = random.sample(li_q, self.K)
                        X_qry = self.tasks['X_all'][kk][query_idx].clone().detach()
                        y_qry = self.tasks['X_all'][kk][query_idx,parm_idx].clone().detach()
                        score_X_qry = self.tasks['score_unconstrainedsamples'][kk][query_idx].clone().detach()

                        _y_pred_qry = fnet.minibatch(X_qry, score_X_qry).detach().squeeze()
                        val_loss = self.criterion(_y_pred_qry,y_qry.squeeze())/self.K + self.weight_decay * _y_pred_qry.detach().pow(2).mean()
                        meta_val_loss += val_loss.detach()/len(meta_val_datasets['X']) # ZS: get the average here

                        # log numbers
                        meta_val_indicator_log[r, kk] = (_y_pred_qry-self.net.c).detach().clone().mean().abs().item() # how close the expectation of Cv close to zero

                        meta_val_loss_log[r, kk] = meta_val_loss.detach().clone()
                        meta_val_truevalues_log[r, kk] =  meta_val_datasets['X_all'][kk][:,parm_idx].mean()

                        meta_val_MC_est_log[r, kk] = y.squeeze().detach().clone().mean()
                        meta_val_MC_2m_est_log[r, kk] = 0.5 * (y_qry.squeeze().detach().clone().mean() + y.squeeze().detach().clone().mean())

                        meta_val_mcv_est_log[r, kk] = (y_qry.squeeze() - _y_pred_qry.detach().squeeze().clone() + self.net.c.detach().clone()).detach().clone().mean()

                        meta_val_abserr_mcv_est_log[r, kk] = (meta_val_mcv_est_log[r, kk] - meta_val_truevalues_log[r, kk]).detach().clone().abs()

                        meta_val_mcv_sim_est_log[r, kk] = self.net.c.detach().clone().squeeze()
                        meta_val_abserr_mcv_sim_est_log[r, kk] = (meta_val_mcv_sim_est_log[r, kk] - meta_val_truevalues_log[r, kk]).detach().clone().abs()

                        meta_val_abserr_MC_est_log[r, kk] = (meta_val_MC_est_log[r, kk] - meta_val_truevalues_log[r, kk]).detach().clone().abs()
                        meta_val_abserr_MC_2m_est_log[r, kk] = (meta_val_MC_2m_est_log[r, kk] - meta_val_truevalues_log[r, kk]).detach().clone().abs()



                self.meta_val_indicator_log_overepoch.append(meta_val_indicator_log.detach().clone()[r,:].mean())
                self.meta_val_loss_log_overepoch.append(meta_val_loss_log.detach().clone()[r,:].mean())
                self.meta_val_abserr_mcv_est_log_overepoch.append(meta_val_abserr_mcv_est_log.detach().clone()[r,:].mean())
                self.meta_val_abserr_mcv_sim_est_log_overepoch.append(meta_val_abserr_mcv_sim_est_log.detach().clone()[r,:].mean())
                self.meta_val_abserr_MC_est_log_overepoch.append(meta_val_abserr_MC_est_log.detach().clone()[r,:].mean())
                self.meta_val_abserr_MC_2m_est_log_overepoch.append(meta_val_abserr_MC_2m_est_log.detach().clone()[r,:].mean())
                if iteration % self.print_every == 0:
                    print(f'{iteration}/{num_iterations} | Meta_Train_Loss: {epoch_loss / self.plot_every:.4f} | Meta_Val_Loss: {meta_val_loss_log[r,:].detach().clone().mean():.4f} | Meta_Val_Indicator:{meta_val_indicator_log[r,:].detach().clone().mean():.4f} | meta_val_abserr_mcv: {meta_val_abserr_mcv_est_log[r,:].detach().clone().mean():.3f} | meta_val_abserr_mcv_sim: {meta_val_abserr_mcv_sim_est_log[r,:].detach().clone().mean():.3f}| meta_val_abserr_MC_est_log_overepoch:{meta_val_abserr_MC_est_log[r,:].detach().clone().mean():.3f} | meta_val_abserr_MC_2m_est_log_overepoch:{meta_val_abserr_MC_2m_est_log[r,:].detach().clone().mean():.3f}')


            if meta_val_datasets is not None:
                self.meta_val_truevalues_log = meta_val_truevalues_log
                self.meta_val_MC_est_log  = meta_val_MC_est_log
                self.meta_val_MC_2m_est_log = meta_val_MC_2m_est_log
                self.meta_val_mcv_est_log = meta_val_mcv_est_log
                self.meta_val_mcv_sim_est_log = meta_val_mcv_sim_est_log

            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0





    def test_storedtasks_rep_lotka_case2(self, meta_test_datasets, ts, te, inner_optim, inner_lr, inner_steps = 1, n_rep = 10, verbose=True):
        parm_idx = self.helper_get_parm()
        self.net.train()  #  higher uses train mode

        inner_opt = inner_optim(self.net.parameters(), lr=inner_lr)

        self.meta_test_loss_log = torch.zeros(n_rep, len(meta_test_datasets['X']))
        self.meta_test_truevalues_log = torch.zeros(n_rep, len(meta_test_datasets['X']))
        self.meta_test_mcv_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))
        self.meta_test_abserr_mcv_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))

        self.meta_test_mcv_sim_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))
        self.meta_test_abserr_mcv_sim_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))

        self.meta_test_MC_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))  # num_iterations
        self.meta_test_MC_2m_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))
        self.meta_test_abserr_MC_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))
        self.meta_test_abserr_MC_2m_est_log = torch.zeros(n_rep, len(meta_test_datasets['X']))


        for r in range(n_rep):
            for kk in range(len(meta_test_datasets['X_all'])):
                li = list(range(0, self.tasks['X_all'][kk].size()[0]))
                support_idx = random.sample(li, self.K)
                X = meta_test_datasets['X_all'][kk][support_idx].clone().detach()
                y = meta_test_datasets['X_all'][kk][support_idx, parm_idx].clone().detach()
                score_X = meta_test_datasets['score_unconstrainedsamples'][kk][support_idx].clone().detach()

                for each_par in self.net.parameters():  ## initialize this neural nets
                    if each_par is self.net.c:
                        torch.nn.init.constant_(each_par, y.detach().mean())


                with higher.innerloop_ctx(self.net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                    # track_higher_grads=False used in test mode
                    for _ in range(inner_steps):
                        _y_pred = fnet.minibatch(X, score_X).squeeze()
                        loss = self.criterion(_y_pred, y.squeeze()) / self.K + self.weight_decay * _y_pred.pow(2).mean()
                        # loss = self.criterion(_y_pred, y.squeeze()) / self.K
                        diffopt.step(loss)
                    # # Get query set
                    li_q = list(set(li)-set(support_idx))
                    query_idx = random.sample(li_q, self.K)
                    X_qry = meta_test_datasets['X_all'][kk][query_idx].clone().detach()
                    y_qry = meta_test_datasets['X_all'][kk][query_idx, parm_idx].clone().detach()
                    score_X_qry = meta_test_datasets['score_unconstrainedsamples'][kk][query_idx].clone().detach()

                    _y_pred_qry = fnet.minibatch(X_qry, score_X_qry).detach().squeeze()
                    val_loss = self.criterion(_y_pred_qry, y_qry.squeeze()).detach()/self.K + self.weight_decay * _y_pred_qry.detach().pow(2).mean()


                    self.meta_test_truevalues_log[r, kk] = meta_test_datasets['X_all'][kk][:,parm_idx].mean()

                    self.meta_test_MC_est_log[r, kk] = y.squeeze().detach().clone().mean()
                    self.meta_test_MC_2m_est_log[r,kk] = 0.5 * (y_qry.squeeze().detach().clone().mean() + y.squeeze().detach().clone().mean())

                    self.meta_test_mcv_est_log[r,kk] = (y_qry.squeeze() - _y_pred_qry.detach() + self.net.c.detach().clone()).mean().detach().clone()
                    self.meta_test_abserr_mcv_est_log[r,kk] = (self.meta_test_mcv_est_log[r,kk] - self.meta_test_truevalues_log[r,kk]).detach().clone().abs()

                    self.meta_test_mcv_sim_est_log[r,kk] = self.net.c.detach().squeeze().clone()
                    self.meta_test_abserr_mcv_sim_est_log[r,kk] = (self.meta_test_mcv_sim_est_log[r,kk] - self.meta_test_truevalues_log[r,kk]).abs().detach().clone()

                    self.meta_test_abserr_MC_est_log[r,kk] = (self.meta_test_MC_est_log[r,kk] - self.meta_test_truevalues_log[r,kk]).detach().clone().abs()
                    self.meta_test_abserr_MC_2m_est_log[r,kk] = (self.meta_test_MC_2m_est_log[r,kk] - self.meta_test_truevalues_log[r,kk]).detach().clone().abs()

            if verbose==True:
                print("SimpCVerr{} | CVerr{} | MCerr{} | MC_2m{}".format(self.meta_test_abserr_mcv_sim_est_log[r,:].mean(), self.meta_test_abserr_mcv_est_log[r,:].mean(), self.meta_test_abserr_MC_est_log[r,:].mean(), self.meta_test_abserr_MC_2m_est_log[r,:].mean()))


        print('Meta_testing finished.')
        self.log_test = []
        self.log_test.append({
            'n_total_tasks': len(meta_test_datasets['X']),
            'loss': self.meta_test_loss_log,
            'True_int_value': self.meta_test_truevalues_log,
            'MC_ests': self.meta_test_MC_est_log,
            'MC_2m_ests': self.meta_test_MC_2m_est_log,
            'CV_ests': self.meta_test_mcv_est_log,
            'Simp_CV_ests': self.meta_test_mcv_sim_est_log,
            'Abserr_CVests': self.meta_test_abserr_mcv_est_log,
            'Abserr_SimpCVests': self.meta_test_abserr_mcv_sim_est_log,
            'Abserr_MCests': self.meta_test_abserr_MC_est_log,
            'Abserr_MC_2m_ests': self.meta_test_abserr_MC_2m_est_log,
            'mode': 'test'
        })
