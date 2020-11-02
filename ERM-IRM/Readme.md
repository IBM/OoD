## Empirical or Invariant Risk Minimization? A Sample Complexity Perspective? 

Instructions on reproducing results from "Empirical or Invariant Risk Minimization? A Sample Complexity Perspective" [link](https://arxiv.org/abs/2010.16412)

Classification Experiments

1. We have attached for Jupyter Notebooks 
	a) Reproduce CS-CMNIST comparisons using IRM_CS_CMNIST_final.ipynb
	b) Reproduce CF-CMNIST comparisons using IRM_CF_CMNIST_final.ipynb
	c) Reproduce AC-CMNIST comparisons using IRM_AC_CMNIST_final.ipynb
	d) Reproduce HB-CMNIST comparisons using IRM_AC_CMNIST_final.ipynb


Regression Experiments

1. Go to IRM_exhaustive_argparse_hper_wo_disc_Sep8.py and enter the path to the directory to save the plots in the already defined variable called dir_name 
2. Run IRM_exhaustive_wo_disc_Sep8.sh and it will generate the plots for the four regression setups (CS-Regression, CF-Regression,AC-Regression, HB-Regression)

Note our code for IRMv1 regression is based on original repository of [IRM](https://github.com/facebookresearch/InvariantRiskMinimization/tree/master/code/experiment_synthetic) which has its own [license](https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/LICENSE)


