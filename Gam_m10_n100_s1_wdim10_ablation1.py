#!/usr/bin/env python

gam = True
M = 10
n = 100
s = 1
wdim = 10
niter = 1000000
ablation = 1

import generic_experiment

generic_experiment.run_experiment(gam,M,n,s,wdim,niter,ablation)
