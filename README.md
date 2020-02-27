# amc-meta-learning-of-optimal-prediction-procedures

Code for paper "Adversarial Monte Carlo Meta-Learning of Optimal Prediction Procedures" by A. Luedtke, I. Cheung, and O. Sofrygin [[link](https://arxiv.org/abs/2002.11275)]

## Environment
All numerical experiments were performed on AWS GPU instances (`p3.2xlarge`), using
- [Python 3.6.5](https://www.python.org/downloads/release/python-365/), and the Python package [Pytorch 1.0.1](pytorch.org/get-started/previous-versions/)

The repository also includes several `R` scripts. All of these scripts were run locally using R 3.5.0 with version 3.2 of the flam package.

## Training new estimators

We trained a total of nine estimators for our paper. The estimators in the linear regression examples were trained using the scripts `Linear_nN_sS_wdim10.py`, where here `N` denotes the sample size at which the estimator is meta-trained, and `S` denotes the sparsity level at which the estimator was meta-trained. For example, the following command can be executed from a bash shell to train our estimator in the linear regression setting using 500 observations at sparsity level 1:

```
./Linear_n500_s1_wdim10.py > Linear_n500_s1_wdim10.out &
```

The estimators for the fused lasso additive model example follow a similar file naming scheme, except `Linear` is replaced by `Gam`. So, for example, the following command can be executed from a bash shell to train our estimator in the fused lasso additive model setting using 500 observations at sparsity level 1:
```
./Gam_n500_s1_wdim10.py > Gam_n500_s1_wdim10.out &
```

We also trained one estimator in the fused lasso additive model example that was not invariant to permutations of the n observations. This non-equivariant estimator was trained by running:
```
./Gam_m10_n100_s1_wdim10_ablation1.py > Gam_m10_n100_s1_wdim10_ablation1.out &
```

## Loading the estimators that we trained

The trained estimators can be found in the `estimators` folder.

Code for loading and applying our estimators to sample data sets can be found in the Jupyter notebook `load_and_apply_estimators.ipynb`.

## Evaluating estimator performance

To evaluate the performance of the learned and existing estimators in linear regression problems, we executed `eval_linear_final.py`. We evaluated the symmetrized etimators in this problem using `eval_linear_final_symmetrized.py`.

To evaluate the performance of the existing FLAM estimator in the fused lasso additive model, we ran the code in the R file `eval_flam.R`. This code generates a folder called `sim_data`, which contains both estimates of the performance of the FLAM estimator and also the data sets used to evaluate this performance. We then evaluated the performance of our unsymmetrized estimators using `eval_flam_final.py`, of our symmetrized estimators using `eval_flam_final_symmetrized.py`, and of our non-equivariant estimators using `eval_flam_ablation.py`.

## Citation
If you use our code, please consider citing the following:

Luedtke A., Cheung I., Sofrygin, O. (2020). Adversarial Monte Carlo Meta-Learning of Optimal Prediction Procedures. arXiv:1712.05835.
