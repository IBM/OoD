"""
Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Last updated: March 12, 2021
Code author: Abhin Shah

File name: vary_dim_size.py

"""

# necessary packages
import numpy as np
import torch
import time
import argparse

from generate_synthetic_data import generate_synthetic_data, generate_E_random
from metrics import PEHE
from models import ERM_Sblock, ERM_Tblock
from irm_block import envs_irm_S, envs_irm_T, IRM_Sblock, IRM_Tblock

def main (args):
  start_time = time.time()
  number_dimension_list = [int(float(item)) for item in args.nd[0].split(',')]

  irm_S_pehe = np.zeros((args.nr, len(number_dimension_list)))
  irm_T_pehe = np.zeros((args.nr, len(number_dimension_list)))
  erm_S_pehe = np.zeros((args.nr, len(number_dimension_list)))
  erm_T_pehe = np.zeros((args.nr, len(number_dimension_list)))

  for r in range(args.nr):
    print("Running iteration: " + str(r+1))
    for j in range(len(number_dimension_list)):
      print("Running dimension: " + str(number_dimension_list[j]))
      args.nd = number_dimension_list[j]
      X_train, T_train, y_train, X_test, T_test, test_potential_outcome = generate_synthetic_data(args)

      # ERM S
      erm_coeff, erm_potential_outcome_S, erm_ate_S = ERM_Sblock(X_train, T_train, X_test, y_train, args.model_type)
      
      # ERM T
      erm_coeff0, erm_coeff1, erm_potential_outcome_T, erm_ate_T = ERM_Tblock(X_train, T_train, X_test, y_train, args.model_type)

      # IRM S
      E = generate_E_random(args.ntr, args.ne)
      environments, featuresTest_control, featuresTest_treatment = envs_irm_S(X_train, X_test, T_train, y_train, E, args.ne)
      irm_coeff, irm_potential_outcome_S, irm_ate_S = IRM_Sblock(environments, featuresTest_control, featuresTest_treatment, args)
      
      # IRM T
      T0_index = np.where(T_train == 0)[0]
      T1_index = np.where(T_train == 1)[0]
      environments_control = envs_irm_T(X_train[T0_index,:], y_train[T0_index,:], E[T0_index,:], args.ne)
      environments_treatment = envs_irm_T(X_train[T1_index,:], y_train[T1_index,:], E[T1_index,:], args.ne)
      features_IRM_T = torch.cat((torch.ones(X_test.shape[0], 1), torch.from_numpy(X_test.astype(np.float32))), 1)
      irm_coeff_c, irm_coeff_t, irm_potential_outcome_T, irm_ate_T = IRM_Tblock(environments_control, environments_treatment, features_IRM_T, args)

      irm_S_pehe[r,j] = PEHE(test_potential_outcome , irm_potential_outcome_S)
      irm_T_pehe[r,j] = PEHE(test_potential_outcome , irm_potential_outcome_T)
      erm_S_pehe[r,j] = PEHE(test_potential_outcome , erm_potential_outcome_S)
      erm_T_pehe[r,j] = PEHE(test_potential_outcome , erm_potential_outcome_T)

      print("--- %s seconds ---" % (time.time() - start_time))

  avg_irm_S_pehe = np.mean(irm_S_pehe, axis=0)
  avg_irm_T_pehe = np.mean(irm_T_pehe, axis=0)
  avg_erm_S_pehe = np.mean(erm_S_pehe, axis=0)
  avg_erm_T_pehe = np.mean(erm_T_pehe, axis=0)

  std_irm_S_pehe = 1/np.sqrt(args.nr) * np.std(irm_S_pehe, axis=0)
  std_irm_T_pehe = 1/np.sqrt(args.nr) * np.std(irm_T_pehe, axis=0)
  std_erm_S_pehe = 1/np.sqrt(args.nr) * np.std(erm_S_pehe, axis=0)
  std_erm_T_pehe = 1/np.sqrt(args.nr) * np.std(erm_T_pehe, axis=0)

  print("ERM_S pehe    " + str(avg_erm_S_pehe) + u" \u00B1 " + str(std_erm_S_pehe))
  print("ERM_T pehe    " + str(avg_erm_T_pehe) + u" \u00B1 " + str(std_erm_T_pehe))
  print("irm_S pehe   " + str(avg_irm_S_pehe) + u" \u00B1 " + str(std_irm_S_pehe))
  print("irm_T pehe   " + str(avg_irm_T_pehe) + u" \u00B1 " + str(std_irm_T_pehe))

  np.savez('feature_model_'+str(args.feature_model)+'_outcome_model_'+str(args.outcome_model)+'_mu_'+str(args.mu)+'_pehe', irm_S_pehe = irm_S_pehe, irm_T_pehe = irm_T_pehe, erm_S_pehe = erm_S_pehe, erm_T_pehe = erm_T_pehe)

  print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--nr',
      help='number of repetitions',
      default=10,
      type=int)
  parser.add_argument(
      '-nd',
      '--item', action='store', nargs='*', dest='nd',
      help='list of dimension size', 
      type=str)
  parser.add_argument(
      '--ne',
      help='number of environments',
      default=2,
      type=int)
  parser.add_argument(
      '--ntr',
      help='number of training observations',
      default=200,
      type=int)
  parser.add_argument(
      '--nte',
      help='number of testing observations',
      default=100,
      type=int)
  parser.add_argument(
      '--mu',
      help='mu',
      default=0.1,
      type=float)
  parser.add_argument(
      '--outcome_model',
      help='outcome_model',
      default="quadratic",
      type=str)
  parser.add_argument(
      '--feature_model',
      help='feature_model',
      default="B",
      type=str)
  parser.add_argument(
      '--sigma_outcome',
      help='sigma_outcome',
      default=1,
      type=int)
  parser.add_argument(
      '--model_type',
      help='linear regression or ridge CV',
      default='RidgeCV',
      type=str)
  parser.add_argument(
      '--number_IRM_iterations',
      help='number of IRM iterations',
      default=10000,
      type=int)
  parser.add_argument(
      '--lr',
      help='IRM lr',
      default=1e-3,
      type=float)
  
  args = parser.parse_args() 
  
  # Calls main function  
  main(args)