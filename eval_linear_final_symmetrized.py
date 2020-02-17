#!/usr/bin/env python

import numpy as np
import torch
import pandas as pd
import os
import copy
from sklearn.linear_model import LinearRegression, LassoCV
from torch.distributions import normal, exponential


import learn2predict as l2p
# Use the device defined in learn2predict (should be the GPU with the most free memory)
device = l2p.device

# Number of distributions to consider
num_mc = 5000
# Number of holdout observations per distribution
n_tilde = 1000
# number of standard methods (ols and lasso)
num_standard = 2

gam = False
M = None
wdim = 10
max_radius = 5

df = pd.DataFrame()

# Error distribution
E = normal.Normal(loc = torch.zeros(1), scale = torch.ones(1))
# Exponential distribution
expo = exponential.Exponential(1.0)

n_metatrain_list = [100,500]

for s in [1,5]:
    T_list = []

    for j in range(len(n_metatrain_list)):
        n_metatrain = n_metatrain_list[j]
        if gam:
            Pi, Pi_opt, Pi_sched = l2p.initPi(s,s+2,wdim,M=M,gam=gam)
            rank_based = True
        else:
            Pi, Pi_opt, Pi_sched = l2p.initPi(s,s+2,wdim,gam=gam)
            rank_based = False

        # initialize the procedure
        T, T_opt, T_sched = l2p.initT(rank_based=rank_based,gam=gam)

        fn_main = './estimators/'+(('Gam' + '_m' + str(M)) if gam else ('Linear')) + '_n' + str(n_metatrain) + '_s' + str(s) + '_wdim' + str(wdim)
        iteration, loss_list = l2p.load_model(T, T_opt, T_sched, Pi, Pi_opt, Pi_sched, fn_main+'.tar', fl_backup = fn_main+'_backup.tar')

        T_list.append(copy.deepcopy(T))

    for scenario in ['interior','boundary']:
        for n in [100,500]:
            print(("n_metatrain:",n_metatrain,"s:",s,"scenario",scenario,"n",n),flush=True)
            losses = [np.zeros(num_mc),np.zeros(num_mc),torch.zeros(num_mc),torch.zeros(num_mc)]
            for i in range(num_mc):
                w_tilde, w, _, _ = Pi(1,n,n_tilde=n_tilde)
                w_tilde = w_tilde.squeeze(0)
                w = w.squeeze(0)
                
                beta = expo.sample([s]).to(device) * (2*(torch.rand(s)>0.5).float()-1).to(device)
                if scenario=='interior':
                    radius = torch.rand(1,device=device)*max_radius
                else:
                    radius = max_radius
                beta = radius * beta/beta.abs().sum(dim=-1,keepdim=True).expand_as(beta)

                if s<wdim:
                    beta = torch.cat((beta,torch.zeros(wdim-s,device=device)),dim=-1)
                # regfun for holdout
                regfun = torch.matmul(w_tilde,beta.unsqueeze(-1)).squeeze(-1)

                # outcome
                y = torch.matmul(w,beta.unsqueeze(-1)) + E.sample([n]).to(device)
                
                # OLS
                ols_preds = LinearRegression().fit(w.cpu(), np.reshape(y.cpu(),-1)).predict(w_tilde.cpu())
                losses[0][i] = ((ols_preds - regfun.cpu().numpy())**2).mean()

                # Lasso
                lasso_preds = LassoCV(cv=10, random_state=0).fit(w.cpu(), np.reshape(y.cpu(),-1)).predict(w_tilde.cpu())
                losses[1][i] = ((lasso_preds - regfun.cpu().numpy())**2).mean()
                
                for j in range(len(n_metatrain_list)):
                    n_metatrain = n_metatrain_list[j]
                    T_out_pos = T_list[j](w_tilde,w,y).squeeze().detach()
                    T_out_neg = -T(w_tilde,w,-y).squeeze().detach()
                    T_out = (T_out_pos + T_out_neg)/2
                    losses[j+num_standard][i] = ((T_out - regfun.squeeze())**2).mean()
            
            
            df = df.append(pd.DataFrame({"scenario":[scenario],
                                         "est":["OLS"],
                                         "n":[np.int(n)],
                                         "s":[np.int(s)],
                                         "mse":[np.float(losses[0].mean())],
                                         "se":[np.sqrt(np.float((losses[0].var()/len(losses[0]))))]}))
            df = df.append(pd.DataFrame({"scenario":[scenario],
                                         "est":["Lasso"],
                                         "n":[np.int(n)],
                                         "s":[np.int(s)],
                                         "mse":[np.float(losses[1].mean())],
                                         "se":[np.sqrt(np.float((losses[1].var()/len(losses[1]))))]}))
            
            for j in range(len(n_metatrain_list)):
                n_metatrain = n_metatrain_list[j]
                df = df.append(pd.DataFrame({"scenario":[scenario],
                                             "est":["AMC"+str(n_metatrain)],
                                             "n":[np.int(n)],
                                             "s":[np.int(s)],
                                             "mse":[np.float(losses[j+num_standard].mean().cpu().numpy())],
                                             "se":[np.sqrt(np.float((losses[j+num_standard].var()/losses[j+num_standard].size()[0]).cpu().numpy()))]}))
    del T_list

df = df.sort_values(by=['s', 'scenario', 'n'])
df.to_csv('tables/linear_results_all_symmetrized.csv', index=False)
df.round(2).to_csv('tables/linear_results_all_symmetrized_rounded.csv', index=False)
