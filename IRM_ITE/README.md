
# Source code for "Treatment Effect Estimation Using Invariant Risk Minimzation"

Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Contact: abhin@mit.edu

Arxiv: [https://arxiv.org/pdf/2103.07788.pdf](https://arxiv.org/pdf/2103.07788.pdf)

### Command inputs:

-   nr: number of repetitions
-   nd: list of dimension size
-   ne: number of environments
-   ntr: number of training observations
-   nte: number of test observations
-   mu: mean used to generate the features
-   outcome_model: outcome generation model (linear or quadratic)
-   feature_model: feature generation model (A or B)
-   sigma_outcome: standard deviation used in outcome generation
-   model_type: model type of ERM (linear regression or ridge regression with cross validation)
-   number_IRM_iterations: number of iterations of IRM
-   lr: learning rate of IRM

### Example command

```shell
$ python3 vary_dim_size.py --nr 10 -nd 5,10,20,35,50 --ne 2 --ntr 200 --nte 100 --mu 0.1 --outcome_model quadratic --feature_model A
$ python3 plotting_dim.py --nr 10 -nd 5,10,20,35,50 --mu 0.1 --outcome_model quadratic --feature_model A
```

