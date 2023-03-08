import torch
import matplotlib.pyplot as plt


MC_Results1 = torch.load('Lotka_Tasks/data_lotka/MC_u1parms.pt')
MCV_Results1 = torch.load('Lotka_Tasks/data_lotka/MCV_u1parms.pt')
MC_Results2 = torch.load('Lotka_Tasks/data_lotka/MC_u2parms.pt')
MCV_Results2 = torch.load('Lotka_Tasks/data_lotka/MCV_u2parms.pt')
NCV_Results1 = torch.load('Lotka_Tasks/data_lotka/Out_ncv_lotka_u1parms.pt')
NCV_Results2 = torch.load('Lotka_Tasks/data_lotka/Out_ncv_lotka_u2parms.pt')


import seaborn as sns
sns.set_theme(style='whitegrid')
ss_lists=[10*2,20*2,30*2,40*2]
parm_id_lists = [0,1,2,3]
plt.figure(figsize=(11, 2 * (5)))
plt.subplot(2, 2, 1)
plt.errorbar(ss_lists, MC_Results1[0,:, 0], 2*MC_Results1[0,:, 1]/20, linestyle='-', marker='o',  color='black',label = "NUTS-MCMC")
plt.errorbar(ss_lists, MCV_Results1[0,:, 0], 2*MCV_Results1[0,:, 1]/20, linestyle='-', marker='o',  color='blue',label = "Meta-CVs")
# plt.errorbar(ss_lists, NCV_Results1[0,:, 0], 2* NCV_Results1[0,:, 1]/20, linestyle='None', marker='o', color='red', label = "NCV")
plt.title(r'$x_1$', size = 24)
plt.ylabel('Mean Absolute Error', size = 24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0)
plt.subplot(2, 2, 2)
plt.errorbar(ss_lists, MC_Results1[1,:, 0], 2*MC_Results1[1,:, 1]/20, linestyle='-', marker='o',   color='black',label = "NUTS-MCMC")
plt.errorbar(ss_lists, MCV_Results1[1,:, 0], 2*MCV_Results1[1,:, 1]/20, linestyle='-', marker='o',  color='blue', label = "Meta-CVs")
# plt.errorbar(ss_lists, NCV_Results1[1,:, 0], 2*NCV_Results1[1,:, 1]/20, linestyle='None', marker='o',  color='red', label = "NCV")
plt.title(r'$x_2$', size = 24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0)
plt.subplot(2, 2, 3)
plt.errorbar(ss_lists, MC_Results2[1,:, 0], 1.96*MC_Results2[1,:, 1]/20, linestyle='-', marker='o',  color='black',label = "NUTS-MCMC")
plt.errorbar(ss_lists, MCV_Results2[1,:, 0], 1.96*MCV_Results2[1,:, 1]/20, linestyle='-', marker='o', color='blue',  label = "Meta-CVs")
# plt.errorbar(ss_lists, NCV_Results2[1,:, 0], 1.96*NCV_Results2[1,:, 1]/20, linestyle='None', marker='o', color='red', label = "NCV")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(r'$x_3$', size = 24)
plt.xlabel('Sample Size: N', size= 24)
plt.ylabel('Mean Absolute Error', size = 24)
plt.ylim(0)

plt.subplot(2, 2, 4)
plt.errorbar(ss_lists, MC_Results2[0,:, 0], 1.96*MC_Results2[0,:, 1]/20, linestyle='-', marker='o',  color='black',label = "NUTS-MCMC")
plt.errorbar(ss_lists, MCV_Results2[0,:, 0], 1.96*MCV_Results2[0,:, 1]/20, linestyle='-', marker='o', color='blue', label = "Meta-CVs")
# plt.errorbar(ss_lists, NCV_Results2[0,:, 0], 1.96*NCV_Results2[0,:, 1]/20, linestyle='None', marker='o', color='red', label = "NCV")
plt.title(r'$x_4$', size = 24)

plt.xlabel('Sample Size: N', size = 24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0)
plt.legend(fontsize=22)
plt.subplots_adjust(wspace=0.4)
plt.tight_layout()

# plt.show()
plt.savefig("lotka.pdf")