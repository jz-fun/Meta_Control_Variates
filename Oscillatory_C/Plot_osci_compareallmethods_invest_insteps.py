import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
# os.getcwd()

# Load data
results_mcv5_invest_insteps = torch.load('Oscillatory_C/data_oscillatory/results_mcv5_invest_insteps_C.pt')

results_CF_d2_results = torch.load('CF/results_CF_d2_results_nonsimCF.pt')

Oscillatory_DF = pd.read_pickle("Oscillatory_C/data_oscillatory/Oscillatory_DF_C.pkl")
results_mcv = torch.tensor([[np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=5')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=5')]['est_abserr'])/np.sqrt(1000)], \
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=10')]['est_abserr']),\
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=10')]['est_abserr'])/np.sqrt(1000)], \
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=20')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=20')]['est_abserr'])/np.sqrt(1000)],\
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=50')]['est_abserr']),\
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=50')]['est_abserr'])/np.sqrt(1000) ],\
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=100')]['est_abserr']),\
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MCV') & (Oscillatory_DF['sample_size'] == 'm=100')]['est_abserr'])/np.sqrt(1000) ]]) # 5,10,20,50,100
results_mc = torch.tensor([[np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=5')]['est_abserr']), \
                            np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=5')]['est_abserr'])/np.sqrt(1000)],\
                           [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=10')]['est_abserr']),\
                            np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=10')]['est_abserr'])/np.sqrt(1000)],\
                           [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=20')]['est_abserr']), \
                            np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=20')]['est_abserr'])/np.sqrt(1000)], \
                           [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=50')]['est_abserr']), \
                            np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=50')]['est_abserr'])/np.sqrt(1000) ], \
                           [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=100')]['est_abserr']), \
                            np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'MC') & (Oscillatory_DF['sample_size'] == 'm=100')]['est_abserr'])/np.sqrt(1000)]])
results_ncv = torch.tensor([[np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=5')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=5')]['est_abserr'])/np.sqrt(1000)], \
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=10')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=10')]['est_abserr'])/np.sqrt(1000)],\
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=20')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=20')]['est_abserr'])/np.sqrt(1000) ], \
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=50')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=50')]['est_abserr'])/np.sqrt(1000)], \
                            [np.mean(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=100')]['est_abserr']), \
                             np.std(Oscillatory_DF.loc[(Oscillatory_DF['method'] == 'NCV') & (Oscillatory_DF['sample_size'] == 'm=100')]['est_abserr'])/np.sqrt(1000)]])




sns.set_style("whitegrid")
plt.figure(figsize=(10, 1 * (5.2)))
plt.subplot(1, 2, 1)
x = [5*2, 10*2, 20*2, 50*2]
plt.errorbar(x[0:4], results_CF_d2_results[0:4,0], 1.96*results_CF_d2_results[0:4, 1], linestyle='-', marker='H', color='grey', label='CF')
plt.errorbar(x[0:4], results_mcv[0:4,0],1.96*results_mcv[0:4, 1] ,linestyle='-', marker='H', color='blue', label='Meta-CVs')
plt.errorbar(x[0:4], results_mc[0:4,0], 1.96*results_mc[0:4, 1], linestyle='-', marker='H', color='black', label ='MC')
plt.errorbar(x[0:4], results_ncv[0:4,0], 1.96*results_ncv[0:4, 1], linestyle='-', marker='H', color='red', label ='Neural-CVs')
plt.legend(fontsize=22)
plt.xlabel('Sample Size: N (d=2)', size = 19)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.ylim(0, 0.3)
plt.ylabel('Mean Absolute Error', size = 22)
plt.subplot(1, 2, 2)
x = [1,3,5,7,10]
plt.errorbar(x[0:len(x)], results_mcv5_invest_insteps[0:len(x),0],1.96*results_mcv5_invest_insteps[0:len(x), 1]/np.sqrt(1000), linestyle='-', marker='H', color='blue', label='Meta-CVs')
plt.errorbar(x[0:len(x)], [results_mc[0,0].detach().numpy()]*len(x), [1.96*results_mc[0,1].detach().numpy()]*len(x), linestyle='-', marker='H', color='black', label ='MC')
plt.errorbar(x[0:len(x)], [results_ncv[0,0].detach().numpy()]*len(x), [1.96*results_ncv[0,1].detach().numpy()]*len(x), linestyle='-', marker='H', color='red',label ='Neural-CVs')
plt.xlabel('L (N=10, d=2)', size = 19)
plt.ylabel('Mean Absolute Error', size = 22)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.ylim(0, 0.3)
plt.tight_layout()
# plt.show()
plt.savefig("Osci_invest_samplesize_and_invest_insteps.pdf")