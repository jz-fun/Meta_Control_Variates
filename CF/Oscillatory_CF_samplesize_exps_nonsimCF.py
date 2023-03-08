from discarded.Oscillatory.Oscillatory_Tasks import *
from CF.sv_CV import *
from CF.utils import *



# Experiments
n_tests = 1000
ss_list =[5,10,20,50]
nss = len(ss_list)
results_CF_d2 = torch.zeros(nss,n_tests,2)
for j in range(0,nss,1):
    K = ss_list[j] # 5 is good with epoch 2000 bs 10; 10 is good with epoch 200 bs 10; 50 is good with (epoch 200 bs 10) ; (all with [80]*2)
    dim=2
    score_dict  = {'SCORE':multivariate_uniform,'dim':dim, 'null_param':0}
    tasks = Oscillatory_Task_Distribution(dim, 4, 6, 0.4, 0.6, 0, 1) # is good when Oscillatory_Task_Distribution(2, 4.5, 5.5, 0.4, 0.6, 0, 1)

    for t in range(n_tests):
        torch.manual_seed(2*(j*n_tests+t)+1)
        task = tasks.sample_task()
        true_val = task.true_integral_val()
        X, Y = task.sample_data(K)
        Score_X = multivariate_uniform(dim, None, X)
        myCF = Simplied_CF(stein_base_kernel_MV_2, rbf_boundcond_multd, X, Y, Score_X)
        myCF.do_tune_kernelparams_negmllk(batch_size_tune=5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=1e-2, epochs=5, verbose=False)

        torch.manual_seed(2 * (j * n_tests + t) + 2)
        Xt, Yt = task.sample_data(K)
        Score_Xt = multivariate_uniform(dim, None, Xt)

        I = myCF.do_nonsim_CF(Xt,Yt, Score_Xt)
        MC_est = Y.mean()
        abs_cf_err = (I - true_val).detach().abs().squeeze()
        abs_mc_err = (MC_est - true_val).detach().abs().squeeze()
        print('{}-{}---mc:{}; cf:{}'.format(j,t,abs_mc_err,abs_cf_err))
        results_CF_d2[j, t, :] = torch.tensor([abs_mc_err, abs_cf_err])


results_CF_d2.mean(1)
results_CF_d2_saved = results_CF_d2[:,:,1]
results_CF_d2_saved.mean(1)

results_CF_d2_saved.size()
results_CF_d2_results = torch.zeros(nss,2)
for i in range(nss):
    results_CF_d2_results[i,0]=results_CF_d2_saved[i].mean()
    results_CF_d2_results[i,1] = results_CF_d2_saved[i].std()/np.sqrt(n_tests)


# Save
# torch.save(results_CF_d2_saved, 'results_CF_d2_saved_nonsimCF.pt')
# torch.save(results_CF_d2_results, 'results_CF_d2_results_nonsimCF.pt')
