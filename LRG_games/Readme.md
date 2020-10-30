

# C-LRG regression


There are two files and one folder.

CLRG_regression_jmtd.ipynb:  Jupyter notebook serves as the tutorial to compare the LRG games methods (described in the [paper](https://arxiv.org/pdf/2010.15234.pdf)) to IRM (Arjovsky et al.), ICP (Peters et al.) and standard ERM for linear regression models. The notebook has comments and explanations to help reproduce the results in the [paper](https://arxiv.org/pdf/2010.15234.pdf).

irm_games_regression.py: This file contains the class fixed_irm_game_model_regression. It has attributes similar to any sklearn class. Initialization requires us to specify the hyper-parameters for the class. fit function is called for training and evaluate function is used for evaluation. (See tutorial notebook for details).

IRMv1_regression: This folder contains slight modification of the regression codes from the [IRM's repository](https://github.com/facebookresearch/InvariantRiskMinimization/tree/master/code/experiment_synthetic) and has its own license.
