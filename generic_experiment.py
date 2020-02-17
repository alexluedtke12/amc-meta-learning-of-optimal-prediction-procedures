####################################################################################
# import libraries

import numpy as np
import pandas as pd
from plotnine import *
import torch
from torch import nn, optim
from torch.distributions import normal, uniform, multivariate_normal, bernoulli, multinomial, chi2
from torch.autograd import Variable
from torch.nn import functional as F
import copy
import os # to check if file exists
from shutil import copyfile # to copy backup file

import learn2predict as l2p

####################################################################################
# Use the device defined in learn2predict (should be the GPU with the most free memory)
device = l2p.device

####################################################################################
# arguments that don't vary across runs

n_tilde = 100 # number of w_tildes (features in simulated holdout set) to use to evaluate risk
nbatch = 100
nT = 1
fixedPi = False

####################################################################################
# set seed
seed = 54321
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

####################################################################################
# ablation=0: not an ablation study
# ablation=1: remove invariance to permutation of observations
def run_experiment(gam,M,n,s,wdim,niter,ablation=0):            
    print("--------------------------------",flush=True)
    print("Fitting "+("generalized additive" if gam else "linear")+" model at n="+str(n)+", s="+str(s)+((", M="+str(M)+",") if gam else (","))+" and wdim="+str(wdim)+".",flush=True)
    
    while True:
        # initialize the prior
        if gam:
            Pi_lr = 0.005
            Pi, Pi_opt, Pi_sched = l2p.initPi(s,s+2,wdim,M=M,gam=gam,lr=Pi_lr)
            rank_based = True
            firstPiIter = 0
        else:
            if s==1:
                Pi_lr = 0.0002
            else:
                Pi_lr = 0.001
            Pi, Pi_opt, Pi_sched = l2p.initPi(s,s+2,wdim,gam=gam,lr=Pi_lr)
            rank_based = False
            firstPiIter = 5000
            
        # initialize the procedure
        if gam:
            T_lr = 0.001
        else:
            if s==1:
                T_lr = 0.0002
            else:
                T_lr = 0.001
        T, T_opt, T_sched = l2p.initT(lr=T_lr,rank_based=rank_based,gam=gam,ablation=ablation)

        fn_main = './estimators/'+(('Gam' + '_m' + str(M)) if gam else ('Linear')) + '_n' + str(n) + '_s'+str(s) + '_wdim' + str(wdim)
        if ablation!=0:
            fn_main = fn_main + '_ablation' + str(ablation)

        try:
            l2p.train(niter,n,n_tilde,nbatch,T,T_opt,T_sched,Pi,Pi_opt,Pi_sched,fn_main+'.tar',loadm=False,new_T_lr=T_lr,new_Pi_lr=Pi_lr,fixedPi=fixedPi,nT=nT,firstPiIter=firstPiIter)
        except l2p.nanError:
            continue
        except Exception as e:
            print(e,flush=True)
        break

    print("--------------------------------",flush=True)
    print("Done.",flush=True)

    return True
