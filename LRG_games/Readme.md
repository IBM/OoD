

# C-LRG regression


There are two files and one folder.

1. CLRG_regression_jmtd.ipynb:  Jupyter notebook serves as the tutorial to compare the LRG games methods (described in the [paper](https://arxiv.org/pdf/2010.15234.pdf)) to IRM (Arjovsky et al.), ICP (Peters et al.) and standard ERM for linear regression models. The notebook has comments and explanations to help reproduce the results in the [paper](https://arxiv.org/pdf/2010.15234.pdf).

2. irm_games_regression.py: This file contains the class fixed_irm_game_model_regression. It has attributes similar to any sklearn class. Initialization requires us to specify the hyper-parameters for the class. fit function is called for training and evaluate function is used for evaluation. (See tutorial notebook for details).

3. IRMv1_regression: This folder contains slight modification of the regression codes from the [IRM's repository](https://github.com/facebookresearch/InvariantRiskMinimization/tree/master/code/experiment_synthetic) and has its own [license](https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/LICENSE).

4. ERM_per_env: This folder is under construction. It contains a simple implementation of a benchmark in which ERM is implemented in every environment and features that do not vary significantly (decided based on a threshold) are declared causal and features that vary more than a threshold are declared non-causal. We retrain the model only on the causal features.

## Suggested Citation

Kartik Ahuja, Karthikeyan Shanmugam, and Amit Dhurandhar, "Linear Regression Games: Convergence Guarantees to Approximate Out-of-Distribution Solutions," arXiv:2010.15234v1, 2020.


    @booklet{AhujaSD2020,
        author="Kartik Ahuja and Karthikeyan Shanmugam and Amit Dhurandhar",
        title="Linear Regression Games: Convergence Guarantees to Approximate Out-of-Distribution Solutions",
        howpublished="arXiv:2010.15234v1",
        year="2020"
    }


