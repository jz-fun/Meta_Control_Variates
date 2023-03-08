from Sarcos.VI_Full_Bayesian_GP_GammaPrior.fb_sarcos_tasks import *

from CF.sv_CV import *
from CF.utils import *





import os
os.getcwd()

X = torch.load('Sarcos/X.pt')
y = torch.load('Sarcos/y.pt')
Xstar = torch.load('Sarcos/Xstar.pt')
Ystar = torch.load('Sarcos/ystar.pt')
X = X.float()
y = y.float()
Xstar = Xstar.float()
Ystar = Ystar.float()

# Get the posterior parameters under Gamma Prior
post_mean_etaparms = torch.tensor([-0.1824,  0.1950])
post_cov_etaparms = torch.tensor([[ 0.0029, -0.0025], [-0.0025,  0.0065]])
score_dict  = {'SCORE':multivariate_Normal_score,'mu':post_mean_etaparms.squeeze().unsqueeze(1), 'cov':post_cov_etaparms}
mte_tasks = fb_Sarcos_Task_Distribution(idces_list = np.arange(1,Xstar.size()[0],2).tolist(), \
                                        X=X, y=y, Xstar=Xstar, Ystar = Ystar, \
                                        post_mean_etaparms=post_mean_etaparms, \
                                        post_cov_etaparms=post_cov_etaparms)



n_tests = 1000
results_CF_sarcos_ss2 = torch.zeros(n_tests,2)

K = 2
for t in range(n_tests):
    torch.manual_seed(2 *  t + 1)
    task = mte_tasks.sample_task()
    true_val = task.true_integral_val()
    X, Y = task.sample_data(K)
    Score_X = multivariate_Normal_score(post_mean_etaparms.squeeze().unsqueeze(1), post_cov_etaparms, X)
    myCF = Simplied_CF(stein_base_kernel_MV_2, rbf_kernel, X, Y, Score_X)
    myCF.do_tune_kernelparams_negmllk(batch_size_tune=2, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=1e-2, epochs=5, verbose=False)

    torch.manual_seed(2 * t )
    Xt, Yt = task.sample_data(K)
    Score_Xt = multivariate_Normal_score(post_mean_etaparms.squeeze().unsqueeze(1), post_cov_etaparms, Xt)
    I = myCF.do_nonsim_CF(Xt, Yt, Score_Xt)

    MC_est = Y.mean()
    abs_cf_err = (I - true_val).detach().abs().squeeze()
    abs_mc_err = (MC_est - true_val).detach().abs().squeeze()
    print('{}---mc:{}; cf:{}'.format(t,abs_mc_err,abs_cf_err))
    results_CF_sarcos_ss2[t, :] = torch.tensor([abs_mc_err, abs_cf_err])

results_CF_sarcos_ss2.size()
results_CF_sarcos_ss2.mean(0)



results_CF_sarcos_ss2 = results_CF_sarcos_ss2[results_CF_sarcos_ss2[:,1] <= 1.75]
plt.boxplot(results_CF_sarcos_ss2[:,1].numpy())
plt.show()

# Save
torch.save(results_CF_sarcos_ss2, 'results_nonsimCF_sarcos_ss2.pt')
