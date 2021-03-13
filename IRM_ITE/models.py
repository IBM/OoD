"""
Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Last updated: March 12, 2021
Code author: Abhin Shah

File name: models.py

Note: ERM model functions
(1) ERM_Sblock: Ordinary least squares/Linear Regression S-learner (OLS/LR1)
(2) ERM_Tblock: Ordinary least squares/Linear Regression T-learner (OLS/LR2)

"""

# necessary packages
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV

def ERM_Sblock(X_train, T_train, X_test, y_train, model_type):
    """Compute the model coefficients, potential outcomes, average treatment effect using OLS/LR1

    Args:
    - X_train: training features
    - T_train: training treatment
    - X_test: test features
    - y_train: training observed outcome
    - model_type: LinearRegression or RidgeCV

    Returns:
    - erm_coeff: the model coefficients of OLS/LR1
    - erm_potential_outcome: computed potential outcomes using OLS/LR1
    - erm_ate: computed average treatment effect using OLS/LR1

    """
    features_train = np.concatenate((X_train, T_train, np.multiply(X_train, T_train)), axis = 1)
    if model_type == 'LinearRegression':
      regressor = LinearRegression()
    elif model_type == 'RidgeCV':
      regressor = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    regressor.fit(features_train, y_train)
    erm_coeff = regressor.coef_

    T_control = np.zeros((X_test.shape[0], 1))
    T_treatment = np.ones((X_test.shape[0], 1))

    features_test_control = np.concatenate((X_test, T_control, np.multiply(X_test, T_control)), axis = 1)    
    features_test_treatment = np.concatenate((X_test, T_treatment, np.multiply(X_test, T_treatment)), axis = 1)

    outcomes_control = regressor.predict(features_test_control)
    outcomes_treatment = regressor.predict(features_test_treatment)

    erm_potential_outcome = np.concatenate((outcomes_control,outcomes_treatment),axis=1)
    erm_individual_effect = erm_potential_outcome[:,1] - erm_potential_outcome[:,0]
    erm_ate = erm_individual_effect.mean()

    return erm_coeff, erm_potential_outcome, erm_ate

def ERM_Tblock(X_train, T_train, X_test, y_train, model_type):
    """Compute the model coefficients, potential outcomes, average treatment effect using OLS/LR2

    Args:
    - X_train: training features
    - T_train: training treatment
    - X_test: test features
    - y_train: training observed outcome
    - model_type: LinearRegression or RidgeCV

    Returns:
    - erm_coeff: the model coefficients of OLS/LR2
    - erm_potential_outcome: computed potential outcomes using OLS/LR2
    - erm_ate: computed average treatment effect using OLS/LR2

    """
    T0_index = np.where(T_train == 0)[0]
    T1_index = np.where(T_train == 1)[0]
    features_test = X_test

    if model_type == 'LinearRegression':
      regressor0 = LinearRegression()
      regressor1 = LinearRegression()
    elif model_type == 'RidgeCV':
      regressor0 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
      regressor1 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

    features_train_0 = X_train[T0_index,:]
    regressor0.fit(features_train_0, y_train[T0_index,:])
    erm_coeff0 = regressor0.coef_
    outcomes_control = regressor0.predict(features_test)

    features_train_1 = X_train[T1_index,:]
    regressor1.fit(features_train_1, y_train[T1_index,:])
    erm_coeff1 = regressor1.coef_
    outcomes_treatment = regressor1.predict(features_test)
    
    erm_potential_outcome = np.concatenate((outcomes_control,outcomes_treatment),axis=1)
    erm_individual_effect = erm_potential_outcome[:,1] - erm_potential_outcome[:,0]
    erm_ate = erm_individual_effect.mean()

    return erm_coeff0, erm_coeff1, erm_potential_outcome, erm_ate