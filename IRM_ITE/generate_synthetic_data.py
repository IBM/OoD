"""
Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Last updated: March 12, 2021
Code author: Abhin Shah

File name: generate_synthetic_data.py

Note: functions to generate the synthetic data
(1) generate_synthetic_data
(2) generate_T
(3) generate_mixture_indicator
(4) generate_E_random
(5) generate_x
(6) generate_outcomes

"""

# necessary packages
import numpy as np
from scipy import random

def generate_synthetic_data(args):
    """Generate the synthetic data for Model B

    Args:
    - args : various parameters of the model

    Returns:
    - X_train: training features
    - T_train: training treatment
    - y_train: training observed outcome
    - X_test: test features
    - test_potential_outcome: test potential outcomes

    """
    number_training_obeservations = args.ntr
    number_testing_obeservations = args.nte
    number_dimensions = args.nd
    mu = args.mu
    feature_model = args.feature_model
    outcome_model = args.outcome_model
    sigma_outcome = args.sigma_outcome
    number_environments = args.ne
    
    T_train = generate_T(number_training_obeservations)
    T_test = generate_T(number_testing_obeservations)

    X_train, X_test = generate_x(number_dimensions, T_train, T_test, mu, feature_model)
    
    train_potential_outcome, test_potential_outcome = generate_outcomes(outcome_model, feature_model, X_train, X_test, sigma_outcome)

    train_po_control = train_potential_outcome[:,0].reshape(number_training_obeservations,1)
    train_po_treatment = train_potential_outcome[:,1].reshape(number_training_obeservations,1)

    y_train = np.multiply(T_train , train_po_treatment) + np.multiply(1-T_train , train_po_control)

    return X_train, T_train, y_train, X_test, T_test, test_potential_outcome

def generate_T(number_obeservations):
    """Generate the treatment variable

    Args:
    - number_obeservations : number of obeservations

    Returns:
    - T: treatment variable

    """
    T = np.random.randint(0, 2, (number_obeservations,1))
    return T

def generate_mixture_indicator(number_obeservations):
    """Generate the variable that will be used for mixture generation

    Args:
    - number_obeservations : number of obeservations

    Returns:
    - mixture_indicator: mixture indicator variable

    """
    mixture_indicator = np.random.randint(0, 2, (number_obeservations,1))
    return mixture_indicator

def generate_E_random(number_obeservations, number_environments):
    """Generate the environment variable randomly

    Args:
    - number_obeservations : number of obeservations
    - number_environments : number of environments

    Returns:
    - E: environment variable

    """
    E = np.random.randint(0, number_environments, (number_obeservations,1))
    return E

def generate_x(number_dimensions, T_train, T_test, mu, feature_model):
    """Generate the features for Model B

    Args:
    - number_dimensions: number of dimensions
    - T_train: training treatment
    - T_test: test treatment
    - mu: mean used in the Gaussian mixture
    - feature_model: feature generation model (A or B)

    Returns:
    - X_train: training features
    - X_test: test features

    """
    number_training_obeservations = T_train.shape[0]
    number_testing_obeservations = T_test.shape[0]

    X_train = np.zeros((number_training_obeservations,number_dimensions))
    X_test = np.zeros((number_testing_obeservations,number_dimensions))

    mixture_indicator_train = generate_mixture_indicator(number_training_obeservations)
    mixture_indicator_test = generate_mixture_indicator(number_testing_obeservations)

    G = np.random.normal(0,1,(number_dimensions,number_dimensions))
    q, r = np.linalg.qr(G)

    mu1 = mu*np.ones(number_dimensions)
    mu2 = -mu*np.ones(number_dimensions)

    if feature_model == "A":
        eigenvalues1 = np.random.uniform(0,1,(number_dimensions,1))
        eigenvalues1 = np.sort(eigenvalues1, axis = 0)/np.sum(eigenvalues1)
        lambda1 = np.identity(number_dimensions)
        np.fill_diagonal(lambda1,eigenvalues1)
        cov1 = q@lambda1@q.T

        for i in range(number_training_obeservations):
            if T_train[i] == 0:
                X_train[i,:] = np.random.multivariate_normal(mu1,cov1,1)
            else:
                X_train[i,:] = np.random.multivariate_normal(mu2,cov1,1)
        
        for i in range(number_testing_obeservations):
            if T_test[i] == 0:
                X_test[i,:] = np.random.multivariate_normal(mu1,cov1,1)
            else:
                X_test[i,:] = np.random.multivariate_normal(mu2,cov1,1)


    elif feature_model == "B":
        eigenvalues1 = np.random.uniform(0,1,(number_dimensions,1))
        eigenvalues1 = np.sort(eigenvalues1, axis = 0)/np.sum(eigenvalues1)
        lambda1 = np.identity(number_dimensions)
        np.fill_diagonal(lambda1,eigenvalues1)
        cov1 = q@lambda1@q.T

        eigenvalues2 = np.random.uniform(0,1,(number_dimensions,1))
        eigenvalues2 = np.sort(eigenvalues2, axis = 0)[::-1]/np.sum(eigenvalues2)
        lambda2 = np.identity(number_dimensions)
        np.fill_diagonal(lambda2,eigenvalues2)
        cov2 = q@lambda2@q.T


        for i in range(number_training_obeservations):
            if T_train[i] == 0:
                if mixture_indicator_train[i] == 0:
                    X_train[i,:] = np.random.multivariate_normal(mu1,cov1,1)
                else:
                    X_train[i,:] = np.random.multivariate_normal(mu1,cov2,1)
            else:
                if mixture_indicator_train[i] == 0:
                    X_train[i,:] = np.random.multivariate_normal(mu2,cov1,1)
                else:
                    X_train[i,:] = np.random.multivariate_normal(mu2,cov2,1)
                    
        for i in range(number_testing_obeservations):
            if T_test[i] == 0:
                if mixture_indicator_test[i] == 0:
                    X_test[i,:] = np.random.multivariate_normal(mu1,cov1,1)
                else:
                    X_test[i,:] = np.random.multivariate_normal(mu1,cov2,1)
            else:
                if mixture_indicator_test[i] == 0:
                    X_test[i,:] = np.random.multivariate_normal(mu2,cov1,1)
                else:
                    X_test[i,:] = np.random.multivariate_normal(mu2,cov2,1)

    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    X_train = (X_train - train_mean)/train_std
    X_test = (X_test - train_mean)/train_std
    
    return X_train, X_test

def generate_outcomes(outcome_model, feature_model, X_train, X_test, sigma_outcome):
    """Generate the potential outcomes for Model B

    Args:
    - outcome_model: decides whether the outcome model is linear or quadratic
    - feature_model: feature generation model (A or B)
    - X_train: training features
    - X_test: test features
    - sigma_outcome: standard deviation of the Gaussian

    Returns:
    - train_potential_outcome: training potential outcomes
    - test_potential_outcome: test potential outcomes

    """
    number_dimensions = X_train.shape[1]
    number_training_obeservations = X_train.shape[0]
    number_testing_obeservations = X_test.shape[0]

    poly_coeff_control_linear = np.random.uniform(0,1,(number_dimensions + 1,1))
    poly_coeff_treatment_linear = np.random.uniform(0,1,(number_dimensions + 1,1))
    
    ones = np.ones((number_training_obeservations,1))
    mean_train_control = np.dot(np.concatenate((ones, X_train), axis = 1) , poly_coeff_control_linear)
    mean_train_treatment = np.dot(np.concatenate((ones, X_train), axis = 1) , poly_coeff_treatment_linear)   

    if outcome_model == "quadratic":
        mean_train_control_quad = np.zeros((number_training_obeservations, 1))
        mean_train_treatment_quad = np.zeros((number_training_obeservations, 1))

        poly_coeff_control_quad = np.random.uniform(0,1,(number_dimensions,number_dimensions))
        poly_coeff_treatment_quad = np.random.uniform(0,1,(number_dimensions,number_dimensions))

        for i in range(number_training_obeservations):
            mean_train_control_quad[i,0] = X_train[i,:] @ poly_coeff_control_quad @ X_train[i,:].T
            mean_train_treatment_quad[i,0] = X_train[i,:] @ poly_coeff_treatment_quad @ X_train[i,:].T

        mean_train_control += mean_train_control_quad
        mean_train_treatment += mean_train_treatment_quad


    sigma_train_potential_outcome = sigma_outcome
    train_potential_outcome_control = np.random.normal(mean_train_control, sigma_train_potential_outcome)
    train_potential_outcome_treatment = np.random.normal(mean_train_treatment, sigma_train_potential_outcome)
    train_potential_outcome = np.concatenate((train_potential_outcome_control,train_potential_outcome_treatment),axis=1)

    ones = np.ones((number_testing_obeservations,1))
    mean_test_control = np.dot(np.concatenate((ones, X_test), axis = 1) , poly_coeff_control_linear)
    mean_test_treatment = np.dot(np.concatenate((ones, X_test), axis = 1) , poly_coeff_treatment_linear)

    if outcome_model == "quadratic":
        mean_test_control_quad = np.zeros((number_testing_obeservations, 1))
        mean_test_treatment_quad = np.zeros((number_testing_obeservations, 1))

        for i in range(number_testing_obeservations):
            mean_test_control_quad[i,0] = X_test[i,:] @ poly_coeff_control_quad @ X_test[i,:].T
            mean_test_treatment_quad[i,0] = X_test[i,:] @ poly_coeff_treatment_quad @ X_test[i,:].T

        mean_test_control += mean_test_control_quad
        mean_test_treatment += mean_test_treatment_quad

    sigma_test_potential_outcome = sigma_outcome
    test_potential_outcome_control = np.random.normal(mean_test_control, sigma_test_potential_outcome)
    test_potential_outcome_treatment = np.random.normal(mean_test_treatment, sigma_test_potential_outcome)
    test_potential_outcome = np.concatenate((test_potential_outcome_control,test_potential_outcome_treatment),axis=1)

    return train_potential_outcome, test_potential_outcome
