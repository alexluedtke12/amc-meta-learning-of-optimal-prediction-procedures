#!/usr/bin/env python

import numpy as np
import torch
import pandas as pd
import os

import learn2predict as l2p
# Use the device defined in learn2predict (should be the GPU with the most free memory)
device = l2p.device

simDir = "./sim_data"
predictor_name = "flam"

num_mc = 2000

gam = True
M = 10
wdim = 10

df = pd.read_csv(os.path.join(simDir,predictor_name+"_performance.csv"),dtype={'scenario':np.int,'est':object,'n':np.int,'s':np.int,'mse':np.float,'se':np.float})

for n_metatrain in [100,500]:
    for s in [1,5]:
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
        for scenario in [1,2,3,4]:
            for n in [100,500]:
                print(("n_metatrain:",n_metatrain,"s:",s,"scenario",scenario,"n",n))
                losses = torch.zeros(num_mc)
                for i in range(num_mc):
                    w = torch.tensor(pd.read_csv(os.path.join(simDir,"flam_"+str(scenario), "w_n"+str(n)+"_s"+str(s)+"_mcrep"+str(i)+".csv")).values,device=device,dtype=torch.float)
                    y = torch.tensor(pd.read_csv(os.path.join(simDir,"flam_"+str(scenario), "y_n"+str(n)+"_s"+str(s)+"_mcrep"+str(i)+".csv")).values,device=device,dtype=torch.float)
                    w_tilde = torch.tensor(pd.read_csv(os.path.join(simDir,"flam_"+str(scenario), "w_tilde_n"+str(n)+"_s"+str(s)+"_mcrep"+str(i)+".csv")).values,device=device,dtype=torch.float)
                    regfun = torch.tensor(pd.read_csv(os.path.join(simDir,"flam_"+str(scenario), "regfun_n"+str(n)+"_s"+str(s)+"_mcrep"+str(i)+".csv")).values,device=device,dtype=torch.float)

                    T_out = T(w_tilde,w,y).squeeze().detach()
                    # T_out_neg = -T(w_tilde,w,-y).squeeze().detach()
                    # T_out_sym = (T_out + T_out_neg)/2

                    # losses[i] = ((T_out_sym - regfun.squeeze())**2).mean()
                    losses[i] = ((T_out - regfun.squeeze())**2).mean()
                    
                df = df.append(pd.DataFrame({"scenario":[np.int(scenario)],
                                             "est":["AMC"+str(n_metatrain)],
                                             "n":[np.int(n)],
                                             "s":[np.int(s)],
                                             "mse":[np.float(losses.mean().cpu().numpy())],
                                             "se":[np.sqrt(np.float((losses.var()/losses.size()[0]).cpu().numpy()))]}))
                
df = df.sort_values(by=['s', 'scenario', 'n'])
df.to_csv('tables/flam_results_all.csv', index=False)
df.round(2).to_csv('tables/flam_results_all_rounded.csv', index=False)
