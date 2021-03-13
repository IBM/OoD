"""
Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Last updated: March 12, 2021
Code author: Abhin Shah

File name: irm_block.py

Note: IRM model functions
(1) envs_irm_S
(2) envs_irm_T
(3) IRM_Sblock
(4) IRM_Tblock

"""

# necessary packages
import numpy as np
import torch
from torch.autograd import grad

def envs_irm_S(X_train, X_test, T_train, y_train, E, number_environments):
    """Compute the environments variable required by the InvariantRiskMinimization class for IRM_S / IRM_1

    Args:
    - X_train: training features
    - X_test: test features
    - T_train: training treatment
    - y_train: training observed outcome
    - E: training environments
    - number_environments: numbe of environments

    Returns:
    - environments: the environments variable required by the InvariantRiskMinimization class for IRM_S / IRM_1
    - featuresTest_control: test features for the control branch of IRM_S / IRM_1
    - featuresTest_treatment: test features for the treatment branch of IRM_S / IRM_1

    """
    X_train = X_train.astype(np.float32)
    T_train = T_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    XT_train =  np.multiply(X_train, T_train)

    environments = []

    for i in range(number_environments):
        E_index = np.where(E == i)[0]
        X = X_train[E_index,:]
        T = T_train[E_index,:]
        y = y_train[E_index,:]
        XT = XT_train[E_index,:]
        X_ = torch.from_numpy(X)
        T_ = torch.from_numpy(T)
        y_ = torch.from_numpy(y)
        XT_ = torch.from_numpy(XT)

        ones_training = torch.ones(X.shape[0], 1)
        features = torch.cat((ones_training, X_, T_, XT_), 1)

        environments.append((features,y_))

    X_test = X_test.astype(np.float32)
    X_test = torch.from_numpy(X_test)
    T_control = torch.zeros(X_test.size()[0], 1)
    T_treatment = torch.ones(X_test.size()[0], 1)
    
    ones_testing = torch.ones(X_test.shape[0], 1)
    featuresTest_control = torch.cat((ones_testing, X_test, T_control, torch.mul(X_test, T_control)), 1)
    featuresTest_treatment = torch.cat((ones_testing, X_test, T_treatment, torch.mul(X_test, T_treatment)), 1)
    
    return environments, featuresTest_control, featuresTest_treatment

def envs_irm_T(X_train, y_train, E, number_environments):
    """Compute the environments variable required by the InvariantRiskMinimization class for IRM_T / IRM_2

    Args:
    - X_train: training features
    - y_train: training observed outcome
    - E: training environments
    - number_environments: numbe of environments

    Returns:
    - environments: the environments variable required by the InvariantRiskMinimization class for IRM_T / IRM_2

    """
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    environments = []

    for i in range(number_environments):
        E_index = np.where(E == i)[0]
        X = X_train[E_index,:]
        y = y_train[E_index,:]
        X_ = torch.from_numpy(X)
        y_ = torch.from_numpy(y)

        ones_training = torch.ones(X.shape[0], 1)
        features = torch.cat((ones_training, X_), 1)
        
        environments.append((features,y_))
    
    return environments


def IRM_Sblock(environments, featuresTest_control, featuresTest_treatment, args):
    """Compute the model coefficients, potential outcomes, average treatment effect using IRM_S / IRM_1

    Args:
    - environments: the environments variable required by the InvariantRiskMinimization class for IRM_S / IRM_1
    - featuresTest_control: test features for the control branch of IRM_S / IRM_1
    - featuresTest_treatment: test features for the treatment branch of IRM_S / IRM_1
    - args: various parameters required for the InvariantRiskMinimization class

    Returns:
    - irm_coeff: the model coefficients of IRM_S / IRM_1
    - irm_potential_outcome: computed potential outcomes using IRM_S / IRM_1
    - irm_ate: computed average treatment effect using IRM_S / IRM_1

    """
    irm = InvariantRiskMinimization(environments, args)  
    irm_coeff = irm.solution()

    outcomes_treatment = featuresTest_treatment @ irm_coeff
    outcomes_control = featuresTest_control @ irm_coeff

    irm_potential_outcome = np.concatenate((outcomes_control.detach().numpy(),outcomes_treatment.detach().numpy()),axis=1)
    irm_individual_effect = irm_potential_outcome[:,1] - irm_potential_outcome[:,0]
    irm_ate = irm_individual_effect.mean()
      
    return irm_coeff, irm_potential_outcome, irm_ate

def IRM_Tblock(environments0, environments1, featuresTest, args):
    """Compute the model coefficients, potential outcomes, average treatment effect using IRM_T / IRM_2

    Args:
    - environments0: the environments variable required by the InvariantRiskMinimization control branch of IRM_T / IRM_2
    - environments1: the environments variable required by the InvariantRiskMinimization treatment branch of IRM_T / IRM_2
    - featuresTest: test features for IRM_T / IRM_2
    - args: various parameters required for the InvariantRiskMinimization class

    Returns:
    - irm_coeff0/irm_coeff1: the model coefficients of IRM_T / IRM_2
    - irm_potential_outcome: computed potential outcomes using IRM_T / IRM_2
    - irm_ate: computed average treatment effect using IRM_T / IRM_2

    """
    irm0 = InvariantRiskMinimization(environments0, args)  
    irm_coeff0 = irm0.solution()
    outcomes_control = featuresTest @ irm_coeff0
    
    irm1 = InvariantRiskMinimization(environments1, args)  
    irm_coeff1 = irm1.solution()
    outcomes_treatment = featuresTest @ irm_coeff1

    irm_potential_outcome = np.concatenate((outcomes_control.detach().numpy(),outcomes_treatment.detach().numpy()),axis=1)
    irm_individual_effect = irm_potential_outcome[:,1] - irm_potential_outcome[:,0]
    irm_ate = irm_individual_effect.mean()
      
    return irm_coeff0, irm_coeff1, irm_potential_outcome, irm_ate


"""
The following class is borrowed from the code repository for Invariant Risk Minimzation -
https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/experiment_synthetic/models.py#L30
"""
class InvariantRiskMinimization(object):
    def __init__(self, environments, args):
        best_reg = 0
        best_err = 1e6
        self.phi = 0

        x_val = environments[-1][0]
        y_val = environments[-1][1]

        for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            self.train(environments[:-1], args, reg=reg)
            err = (x_val @ self.solution() - y_val).pow(2).mean().item()

            if err < best_err:
                best_err = err
                best_reg = reg
                best_phi = self.phi.clone()

        self.phi = best_phi

    def train(self, environments, args, reg=0):
        dim_x = environments[0][0].size(1)

        self.phi = torch.nn.Parameter(torch.eye(dim_x, dim_x))
        self.w = torch.ones(dim_x, 1)
        self.w.requires_grad = True

        opt = torch.optim.Adam([self.phi], lr=args.lr)
        loss = torch.nn.MSELoss()

        for iteration in range(args.number_IRM_iterations):
            penalty = 0
            error = 0
            for x_e, y_e in environments:
                error_e = loss(x_e @ self.phi @ self.w, y_e)
                penalty += grad(error_e, self.w,
                                create_graph=True)[0].pow(2).mean()
                error += error_e

            opt.zero_grad()
            (reg * error + (1 - reg) * penalty).backward()
            opt.step()


    def solution(self):
        return self.phi @ self.w
        