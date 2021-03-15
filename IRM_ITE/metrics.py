"""
Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Last updated: March 12, 2021
Code author: Abhin Shah

File name: metrics.py

Note: metric functions
(1) PEHE: Precision in Estimation of Heterogeneous Effect
(2) ATE: Average Treatment Effect

"""

# necessary packages
import numpy as np

def PEHE(test_potential_outcome , pred_potential_outcome):
    """Compute Precision in Estimation of Heterogeneous Effect.

    Args:
    - test_potential_outcome: true potential outcomes
    - pred_potential_outcome: estimated potential outcomes

    Returns:
    - pehe: PEHE

    """
    ite_test = test_potential_outcome[:,1] - test_potential_outcome[:,0]
    ite_pred = pred_potential_outcome[:,1] - pred_potential_outcome[:,0]
    pehe = np.mean(np.square(ite_test - ite_pred), axis=0)
    return np.sqrt(pehe)

def ate_error(test_ate, pred_ate):
    """Compute the error in Average Treatment Effect.

    Args:
    - test_ate: true average treatment effect
    - pred_ate: estimated average treatment effect

    Returns:
    - ate_error: computed error in average treatment effect
    
    """
    ate_error = np.abs(test_ate - pred_ate)
    return ate_error