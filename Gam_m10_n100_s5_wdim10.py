#!/usr/bin/env python

gam = True
M = 10
n = 100
s = 5
wdim = 10
niter = 1000000

import generic_experiment

generic_experiment.run_experiment(gam,M,n,s,wdim,niter)
