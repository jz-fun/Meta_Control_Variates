
import torch
import numpy as np

import stan
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
from scipy import integrate



##### Lotka Stan code
lotka_code = """
functions {
  real[] dz_dt(real t, // time
                     array[] real z,
                     // system state {prey, predator}
                     array[] real theta, // parameters
                     array[] real x_r, // unused data
                     array[] int x_i) {
    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta = theta[2];
    real gamma = theta[3];
    real delta = theta[4];

    real du_dt = (exp(alpha) - exp(beta) * v) * u;
    real dv_dt = (-exp(gamma) + exp(delta) * u) * v;
    return {du_dt, dv_dt};
  }
}
data {
  int<lower = 0> N;           // num measurements
  real ts[N];                 // measurement times > 0
  real<lower=0> y_init[2];             // initial measured population
  real<lower = 0> y[N, 2];    // measured population at measurement times
  int<lower = 0> N_zgen;
  real ts_N_zgen[N_zgen]; 
}
parameters {
  real theta[4];   // theta = { alpha, beta, gamma, delta }
  real z_init[2];  // initial population
  real sigma[2];   // error scale
}
transformed parameters {
  real z[N, 2]
    = integrate_ode_rk45(dz_dt, exp(z_init), 0, ts, theta,
                         rep_array(0.0, 0), rep_array(0, 0),
                         1e-6, 1e-5, 1e3);
}
model {
  theta[{1, 3}] ~ normal(0, 0.5);
  theta[{2, 4}] ~ normal(-3, 0.5);
  sigma ~ normal(-1, 1);
  z_init ~ normal(log(10), 1);
  for (k in 1:2) {
    y_init[k] ~ lognormal(log(exp(z_init[k])), exp(sigma[k]));
    y[ , k] ~ lognormal(log(z[, k]), exp(sigma[k]));
  }
}
generated quantities {
  real z_gen[N_zgen, 2]
    = integrate_ode_rk45(dz_dt, exp(z_init), 0, ts_N_zgen, theta,
                         rep_array(0.0, 0), rep_array(0, 0),
                         1e-6, 1e-5, 1e3);
  array[2] real y_init_rep;
  array[N, 2] real y_rep;
  for (k in 1 : 2) {
    y_init_rep[k] = lognormal_rng(log(exp(z_init[k])), exp(sigma[k]));
    for (n in 1 : N) {
      y_rep[n, k] = lognormal_rng(log(z[n, k]), exp(sigma[k]));
    }
  }
}
"""





# Helper function to get u : u = f(a, b,c,d, u_0)
def func_u( a,b,c,d, u0 = np.array([33.68, 5.92]),t=np.linspace(0, 25, 26)):

    def dX_dt(X, t=0):
        """ Return the growth rate of fox and rabbit populations. """
        return np.array([np.exp(a) * X[0] - np.exp(b) * X[0] * X[1], -np.exp(c) * X[1] + np.exp(d) * X[0] * X[1]])

    X, infodict = integrate.odeint(dX_dt, u0, t, full_output=True)
    hare, lynx = X.T

    # 2d torch tensor including u_0 in the first row
    u_df = torch.stack((torch.tensor(hare), torch.tensor(lynx)), 1)
    return u_df.detach()





#
class Lotka_Task():
    def __init__(self, lynx_hare_data):
        self.lynx_hare_data = lynx_hare_data

    def sample_observations(self, size=1):
        """
         Sample a subset of all observations each time
        :param size: sample size of the subset
        :return: a dict like self.lynx_hare_data
        """
        N = self.lynx_hare_data['N']
        y_tensor = torch.tensor(self.lynx_hare_data['y'])


        idx = torch.randperm(N)
        ts_sub_raw = idx[0:size].tolist()
        ts_sub =(1+idx[0:size]).tolist() # add 1 is because of this need to be +1 in Def. of ode solver: only the initial point index is 0
        ts_sub_sorted = np.sort(ts_sub).tolist()
        # y_sublist_sorted = y_tensor[torch.tensor(np.sort(ts_sub_raw).tolist())].tolist()
        y_sublist_sorted = y_tensor[np.sort(ts_sub_raw).tolist()].tolist()

        subdatadict  = {'N': size, 'ts': ts_sub_sorted,\
                        'y_init': self.lynx_hare_data['y_init'], 'y': y_sublist_sorted, \
                        'N_zgen': self.lynx_hare_data['N_zgen'], 'ts_N_zgen': self.lynx_hare_data['ts_N_zgen']}

        return subdatadict



    def post_inference(self, X_dict, nchains, nsams_perchain):
        """
        Each time compile a Stan model and return the evaluations of the target integrand (ann posterior samples)
        https://discourse.mc-stan.org/t/can-a-pystan-3-model-be-compiled-without-data/23497

        :param X_dict: a dict; contain info to fit the stan model; note that observations are a subset of all
        :param nchains: e.g.4
        :param nsams_perchain: e.g.100
        :return:  evaluations of u(s) given X_dict ; posterior samples of parameters given X_dict observations; the corresponding scores
        """
        #  Build model and sample
        posterior = stan.build(program_code=lotka_code, data=X_dict)
        fit = posterior.sample(num_chains=nchains, num_samples=nsams_perchain)

        z_init = fit['z_init']
        sigma = fit['sigma']
        theta = fit["theta"]
        score_unconstrainedsamples = torch.zeros(nchains * nsams_perchain, 8)
        for i in range(nchains * nsams_perchain):
            #  function 'unconstrain_pars' takes in constrained values and returns the corresponding unconstrained values.
            #         then the function 'grad_log_prob' will calculate the gradient of log posterior at the corresponing unconstrained values.
            temp = posterior.grad_log_prob(posterior.unconstrain_pars(dict(theta=theta[:, i].tolist(), \
                                                                           z_init=z_init[:, i].tolist(), \
                                                                           sigma=sigma[:, i].tolist())))
            score_unconstrainedsamples[i] = torch.tensor(temp)

        z_init_tensor = torch.tensor(z_init)  # 2d tensor of size [2, num_chains*num_samples]
        sigma_tensor = torch.tensor(sigma)  # 2d tensor of size [2, num_chains*num_samples]
        theta_tensor = torch.tensor(theta)  # 2d tensor of size [4, num_chains*num_samples]
        y_rep = torch.tensor(fit['y_rep'])
        y_init_rep = torch.tensor(fit['y_init_rep'])

        # Save data
        X_all = torch.cat((theta_tensor, z_init_tensor, sigma_tensor), 0)
        X_all = X_all.t().float() # 2d tensor of size [num_chains*num_samples, 8]
        z_rep_tensor = torch.tensor(fit['z_gen']).float() # 3d tensor of size [N', 2, num_chains*num_samples], the last dim equals the number of posterior samples

        return z_rep_tensor, X_all, score_unconstrainedsamples, y_rep, y_init_rep




    def true_z_vals(self, X_dict, nchains=5, nsams_perchain=10000):
        """
        :return: z_rep_tensor.mean(dim=2) : 2d tensor of size [N',2]
                 z_rep_tensor: 3d tensor of size [N', 2, num_chains*num_samples],
                 X_all: 2d tensor of size [num_chains*num_samples, 8]
                 score_unconstrainedsamples: 2d tensor of size [num_chains*num_samples, 8]
        """
        self.z_rep_tensor, self.X_all, self.score_unconstrainedsamples , self.y_rep, self.y_init_rep= self.post_inference(X_dict, nchains=nchains,\
                                                                                            nsams_perchain=nsams_perchain)

        return self.z_rep_tensor.mean(dim=2), self.z_rep_tensor, self.X_all, self.score_unconstrainedsamples, self.y_rep, self.y_init_rep




    def func_u_given_postsamples(self, x):
        pass







class Lotka_Task_Distribution():
    def __init__(self, lynx_hare_data):
        self.lynx_hare_data = lynx_hare_data

    def sample_task(self, num_obs, num_s, num_q, nchains,nsams_perchain):
        task = Lotka_Task(self.lynx_hare_data)
        data_dict = task.sample_observations(size = num_obs)
        true_vals, z_rep_tensor, X_all, score_unconstrainedsamples, y_rep, y_init_rep= task.true_z_vals(X_dict = data_dict,\
                                                                                               nchains=nchains, \
                                                                                               nsams_perchain=nsams_perchain)

        X = X_all[0:num_s+num_q, :]
        score_X = score_unconstrainedsamples[0:num_s+num_q, :]
        FX = z_rep_tensor[:, :,  0:num_s+num_q]

        # 'z_rep_tensor, X_all, score_unconstrainedsamples': record all sampled samples and evaluations of integrands and score function
        # For meta CV computation only 'true_vals, X, score_X, FX' is needed
        return true_vals, X, score_X, FX, data_dict, y_rep, y_init_rep, z_rep_tensor, X_all, score_unconstrainedsamples







def generate_meta_data(lynx_hare_data, numtasks, n_s, n_q, num_chains_true_intval, num_samperchain_true_intval,ifuse_allobs, obs_size = 5):
        if ifuse_allobs == True:
            # When 'ifuse_allobs' is true, that means we use all observations to get the posterior; each task just use different posteriors
            # This means the model only compile once and sample once for all tasks. Compile once because the stan code of model doesnot change.
            assert obs_size == None, 'In this case, obs_size equal to full obs size; therefore no meaning to define a value here.'
            # assert n_s == None, 'In this case, n_s equal to full obs size divided by 2 as all samples for all tasks are from the same posetrior;' \
            #                     ' and samples for each task will then be sampled from the stored giant dataset; therefore no meaning to define a value here.'
            # assert n_q == None, 'In this case, n_q equal to full obs size divided by 2 as all samples for all tasks are from the same posetrior;' \
            #                     ' and samples for each task will then be sampled from the stored giant dataset; therefore no meaning to define a value here.'
            # assert numtasks == None, 'In this case, all samples for all tasks are from the same posetrior;' \
            #                     ' and samples for each task will then be sampled from the stored giant dataset; therefore no meaning to define a value here.'
            meta_taskdist = Lotka_Task_Distribution(lynx_hare_data)
            total_ss = num_chains_true_intval * num_samperchain_true_intval
            true_vals, X, score_X, FX, data_dict, y_rep, y_init_rep, z_rep_tensor, X_all, score_unconstrainedsamples_all \
                = meta_taskdist.sample_task(num_obs=lynx_hare_data['N'], \
                                            num_s=total_ss//2, num_q=total_ss//2, \
                                            nchains=num_chains_true_intval,\
                                            nsams_perchain=num_samperchain_true_intval)

            assert numtasks * (n_s + n_q) < num_chains_true_intval * num_samperchain_true_intval, 'Samples are not enough to create the defined number of tasks.'
            nt = n_s+n_q
            tvl, Xl, score_Xl, FXl = [],[],[],[]
            for i in range(numtasks):
                tvl.append(true_vals[i*nt:(i+1)*nt])
                Xl.append(X[i*nt:(i+1)*nt])
                score_Xl.append(score_X[i*nt:(i+1)*nt])
                FXl.append(FX[:, :, i*nt:(i+1)*nt])

            metadata = {"true_vals": tvl, 'X':Xl, 'score_X': score_Xl, 'FX':FXl, \
                        'data_dict':data_dict, 'y_rep':y_rep, 'y_init_rep':y_init_rep, 'z_rep_tensor':z_rep_tensor,\
                        'X_all': X_all, 'score_unconstrainedsamples_all':score_unconstrainedsamples_all}
            return metadata

        elif ifuse_allobs == False:
            # When 'ifuse_allobs' is false, that means for each task, we will have different posteriors.
            # This means the model compile once and sample once for each task. Compile once because the stan code of model doesnot change.
            meta_true_vals, meta_X, meta_score_X, meta_FX = [], [],[],[]
            meta_data_dict, meta_y_rep, meta_y_init_rep = [],[],[]
            meta_z_rep_tensor, meta_X_all, meta_score_unconstrainedsamples = [], [], []

            meta_taskdist = Lotka_Task_Distribution(lynx_hare_data)
            for i in range(numtasks):
                print('Current {}th task/{}tasks'.format(i+1, numtasks))
                true_vals, X, score_X, FX, data_dict, y_rep, y_init_rep, z_rep_tensor, X_all, score_unconstrainedsamples = meta_taskdist.sample_task(\
                    num_obs=obs_size, \
                    num_s=n_s, num_q=n_q, \
                    nchains=num_chains_true_intval, \
                    nsams_perchain=num_samperchain_true_intval)

                meta_true_vals.append(true_vals)
                meta_X.append(X)
                meta_score_X.append(score_X)
                meta_FX.append(FX)

                meta_data_dict.append(data_dict)
                meta_y_rep.append(y_rep)
                meta_y_init_rep.append(y_init_rep)
                meta_z_rep_tensor.append(z_rep_tensor)

                meta_X_all.append(X_all)
                meta_score_unconstrainedsamples.append(score_unconstrainedsamples)

            metadata = {"true_vals": meta_true_vals, 'X': meta_X, 'score_X': meta_score_X, 'FX': meta_FX, 'data_dict': meta_data_dict,
                        'y_rep': meta_y_rep, 'y_init_rep': meta_y_init_rep, 'z_rep_tensor': meta_z_rep_tensor, 'X_all':meta_X_all,'score_unconstrainedsamples':meta_score_unconstrainedsamples}
            print('Finished!')
            return metadata





# Generate Meta Dtatsets for Meta CV
import os
print("Path at terminal when executing this file")
print(os.getcwd() + "\n")
lynx_hare_df = pd.read_csv("hudson-bay-lynx-hare.csv", header=2)
N = len(lynx_hare_df) - 1
ts = np.arange(1, N + 1, 1).tolist()
y_init = [lynx_hare_df.iloc[0, 2], lynx_hare_df.iloc[0, 1]]  # c(lynx_hare_df$Hare[1], lynx_hare_df$Lynx[1])
y = np.array(lynx_hare_df.iloc[1:(N + 1), [2, 1]])  # hare, lynx order
y = y.tolist()
ts_N_zgen = np.arange(1, N + 1, 0.2).tolist()
N_zgen = len(ts_N_zgen)
lynx_hare_data = {'N': N, 'ts': ts, 'y_init': y_init, 'y': y, 'N_zgen': N_zgen, 'ts_N_zgen': ts_N_zgen}



# # To generate data, use the following code:
# Meta Training Data
metadata = generate_meta_data(lynx_hare_data=lynx_hare_data, numtasks=100, n_s=5, n_q=5, num_chains_true_intval=4, num_samperchain_true_intval=1000,ifuse_allobs=False, obs_size=15)
torch.save(metadata, 'metadata_lotka.pt')
#
# # Meta Test Data
metatestdata = generate_meta_data(lynx_hare_data=lynx_hare_data, numtasks=40, n_s=5, n_q=5, num_chains_true_intval=4, num_samperchain_true_intval=1000,ifuse_allobs=False, obs_size=15)
torch.save(metatestdata, 'metatestdata_lotka.pt')


# Combinatory number -- total number of available tasks
# from scipy.special import comb, perm
# comb(20,15)



if __name__ =="__main__":
    # Read Observations and Define Generated Quantities
    lynx_hare_df = pd.read_csv("hudson-bay-lynx-hare.csv", header=2)
    N = len(lynx_hare_df) - 1
    ts = np.arange(1, N + 1, 1).tolist()
    y_init = [lynx_hare_df.iloc[0, 2], lynx_hare_df.iloc[0, 1]]  # c(lynx_hare_df$Hare[1], lynx_hare_df$Lynx[1])
    y = np.array(lynx_hare_df.iloc[1:(N + 1), [2, 1]])  # hare, lynx order
    y = y.tolist()

    ts_N_zgen = np.arange(1, N + 1, 0.2).tolist()
    N_zgen = len(ts_N_zgen)

    lynx_hare_data = {'N': N, 'ts': ts, 'y_init': y_init, 'y': y, 'N_zgen': N_zgen, 'ts_N_zgen': ts_N_zgen}


    # Test a single task generated from meta env
    metaenv = Lotka_Task_Distribution(lynx_hare_data)
    tv, x, sx, fx, ddct, y_rep_tensor, y_init_rep_tensor, z_rep_tensor, x_all, sx_all = metaenv.sample_task(5, 2, 2, 4, 20)
    # Only "tv, x, sx, fx" are used for CVs; the other are used for visualisation


    plt.figure()
    for i in range(8):
        plt.subplot(3, 3, i + 1)
        plt.hist(x_all[:,i].numpy(), bins=10)
    plt.show()

    mean_ppc = y_rep_tensor.mean(dim=2)
    CriL_ppc = torch.quantile(y_rep_tensor, q=0.025, dim=2)
    CriU_ppc = torch.quantile(y_rep_tensor, q=0.975, dim=2)
    # ddct['ts']
    Year = np.array(ddct['ts'])+1901
    y = ddct['y']
    plt.figure(figsize=(15, 2 * (5)))
    plt.subplot(2, 2, 1)
    plt.plot(Year, torch.tensor(y)[:, 1].numpy(), "o", color="b", lw=4, ms=10.5)  # 0 is hare, 1 is lynx
    plt.plot(Year, mean_ppc[:, 1].numpy(), color="b", lw=4)
    plt.plot(Year, CriL_ppc[:, 1].numpy(), "--", color="b", lw=2)
    plt.plot(Year, CriU_ppc[:, 1].numpy(), "--", color="b", lw=2)
    plt.xlim([1900, 1922])
    plt.ylim([0, 100])
    plt.ylabel("Lynx", fontsize=15)
    plt.xticks(Year, rotation=45, fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(2, 2, 2)
    plt.plot(Year, torch.tensor(y)[:, 0].numpy(), "o", color="g", lw=4, ms=10.5, label="Observed")
    plt.plot(Year, mean_ppc[:, 0].numpy(), color="g", lw=4, label="mean of ppc")
    plt.plot(Year, CriL_ppc[:, 0].numpy(), "--", color="g", lw=2, label="credible intervals")
    plt.plot(Year, CriU_ppc[:, 0].numpy(), "--", color="g", lw=2)
    plt.legend(fontsize=15)
    plt.xlim([1900, 1922])
    plt.ylim([0, 120])
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Hare", fontsize=15)
    plt.xticks(Year, rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    # plt.show()

    ### ------- z_gen ---
    # z_rep_tensor.size()
    # 3d tensor of size [N, 2, num_chains*num_samples], the last dim equals the number of posterior samples

    Year = np.arange(1901, 1920 + 1, 0.2)

    mean_ppc = z_rep_tensor.mean(dim=2)
    CriL_ppc = torch.quantile(z_rep_tensor, q=0.025, dim=2)
    CriU_ppc = torch.quantile(z_rep_tensor, q=0.975, dim=2)

    # plt.figure(figsize=(15, 2 * (5)))
    plt.subplot(2, 2, 3)
    plt.plot(Year, mean_ppc[:, 1].numpy(), color="b", lw=4)
    plt.plot(Year, CriL_ppc[:, 1].numpy(), "--", color="b", lw=2)
    plt.plot(Year, CriU_ppc[:, 1].numpy(), "--", color="b", lw=2)
    plt.xlim([1900, 1922])
    plt.ylim([0, 80])
    plt.ylabel("Lynx", fontsize=15)
    plt.xticks(range(int(min(Year.tolist())), int(max(Year)+1)), rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplot(2, 2, 4)
    plt.plot(Year, mean_ppc[:, 0].numpy(), color="g", lw=4, label="mean of ppc")
    plt.plot(Year, CriL_ppc[:, 0].numpy(), "--", color="g", lw=2, label="credible intervals")
    plt.plot(Year, CriU_ppc[:, 0].numpy(), "--", color="g", lw=2)
    plt.legend(fontsize=15)
    plt.xlim([1900, 1922])
    plt.ylim([0, 120])
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Hare", fontsize=15)
    plt.xticks(range(int(min(Year.tolist())), int(max(Year)+1)), rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


