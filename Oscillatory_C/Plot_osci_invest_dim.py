import torch
import numpy as np
import matplotlib.pyplot as plt


results_mcv5 = torch.load('Oscillatory_C/data_oscillatory/results_mcv5_C.pt')
results_mc5 = torch.load('Oscillatory_C/data_oscillatory/results_mc5_C.pt')
results_ncv5 = torch.load('Oscillatory_C/data_oscillatory/results_ncv5_C.pt')
results_mcv20 = torch.load('Oscillatory_C/data_oscillatory/results_mcv20_C.pt')
results_mc20 = torch.load('Oscillatory_C/data_oscillatory/results_mc20_C.pt')
results_ncv20 = torch.load('Oscillatory_C/data_oscillatory/results_ncv20_C.pt')
results_CF5_results = torch.load('CF/results_CF5_results_nonsimCF.pt')
results_CF20_results = torch.load('CF/results_CF20_results_nonsimCF.pt')
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(10, 1 * (5.2)))
plt.subplot(1, 2, 2)
x = [1, 2,3,4,5,6, 7]
plt.errorbar(x[0:7], results_CF20_results[0:7,0],1.96*results_CF20_results[0:7, 1], linestyle='-', marker='H', color='grey', label='CF')
plt.errorbar(x[0:7], results_mcv20[0:7,0],1.96*results_mcv20[0:7, 1]/np.sqrt(1000), linestyle='-', marker='H', color='blue', label='Meta-CVs')
plt.errorbar(x[0:7], results_mc20[0:7,0], 1.96*results_mc20[0:7, 1]/np.sqrt(1000), linestyle='-', marker='H', color='black', label ='MC')
plt.errorbar(x[0:7], results_ncv20[0:7,0], 1.96*results_ncv20[0:7, 1]/np.sqrt(1000), linestyle='-', marker='H', color='red', label ='Neural-CVs')
plt.legend(fontsize=22)
plt.xlabel('Num. of Dim. d (N=20, L=1)', size = 18)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.ylim(0.)

plt.subplot(1, 2, 1)
x = [1, 2,3,4,5,6, 7]
plt.errorbar(x[0:7], results_CF5_results[0:7,0],1.96*results_CF5_results[0:7, 1], linestyle='-', marker='H', color='grey', label='CF')
plt.errorbar(x[0:7], results_mcv5[0:7,0], 1.96*results_mcv5[0:7, 1]/np.sqrt(1000), linestyle='-', marker='H', color='blue', label='Meta-CVs') #capsize=5,
plt.errorbar(x[0:7], results_mc5[0:7,0], 1.96*results_mc5[0:7, 1]/np.sqrt(1000), linestyle='-', marker='H', color='black', label ='MC')
plt.errorbar(x[0:7], results_ncv5[0:7,0], 1.96*results_ncv5[0:7, 1]/np.sqrt(1000), linestyle='-', marker='H', color='red',label ='Neural-CVs')
plt.xlabel('Num. of Dim. d (N=5, L=1)', size = 18)
plt.ylabel('Mean Absolute Error' ,size = 22)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.ylim(0.)
plt.tight_layout()
# plt.show()
plt.savefig("Osci_invest_dim.pdf")