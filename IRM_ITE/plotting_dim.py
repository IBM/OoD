"""
Reference: Abhin Shah, Kartik Ahuja, Karthikeyan Shanmugam, Dennis Wei, Kush Varshney, Amit Dhurandhar,
"Treatment Effect Estimation using Invariant Risk Minimization," 
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021

Last updated: March 12, 2021
Code author: Abhin Shah

File name: plotting_dim.py

"""

# necessary packages
import numpy as np
import argparse
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def main (args):
  number_dimension_list = [int(float(item)) for item in args.nd[0].split(',')]

  npzfile = np.load('feature_model_'+str(args.feature_model)+'_outcome_model_'+str(args.outcome_model)+'_mu_'+str(args.mu)+'_pehe.npz')

  irm_S_pehe = npzfile['irm_S_pehe']
  irm_T_pehe = npzfile['irm_T_pehe']
  erm_S_pehe = npzfile['erm_S_pehe']
  erm_T_pehe = npzfile['erm_T_pehe']

  avg_irm_S_pehe = np.mean(irm_S_pehe, axis=0)
  avg_irm_T_pehe = np.mean(irm_T_pehe, axis=0)
  avg_erm_S_pehe = np.mean(erm_S_pehe, axis=0)
  avg_erm_T_pehe = np.mean(erm_T_pehe, axis=0)

  std_irm_S_pehe = 1/np.sqrt(args.nr) * np.std(irm_S_pehe, axis=0)
  std_irm_T_pehe = 1/np.sqrt(args.nr) * np.std(irm_T_pehe, axis=0)
  std_erm_S_pehe = 1/np.sqrt(args.nr) * np.std(erm_S_pehe, axis=0)
  std_erm_T_pehe = 1/np.sqrt(args.nr) * np.std(erm_T_pehe, axis=0)

  # print("ERM_S pehe    " + str(avg_erm_S_pehe) + u" \u00B1 " + str(std_erm_S_pehe))
  # print("ERM_T pehe    " + str(avg_erm_T_pehe) + u" \u00B1 " + str(std_erm_T_pehe))
  # print("irm_S pehe   " + str(avg_irm_S_pehe) + u" \u00B1 " + str(std_irm_S_pehe))
  # print("irm_T pehe   " + str(avg_irm_T_pehe) + u" \u00B1 " + str(std_irm_T_pehe))

  darkblue = mlines.Line2D([], [], color='darkblue', marker='<', linestyle='None', markersize=5, label=r'IRM$_1$')
  orange = mlines.Line2D([], [], color='orange', marker='>', linestyle='None', markersize=5, label=r'IRM$_2$')
  green = mlines.Line2D([], [], color='green', marker='v', linestyle='None', markersize=5, label=r'OLS/LR1')
  red = mlines.Line2D([], [], color='red', marker='^', linestyle='None', markersize=5, label=r'OLS/LR2')

  plt.figure()
  plt.errorbar(number_dimension_list, avg_irm_S_pehe, std_irm_S_pehe, color='darkblue', marker='<', linestyle='dotted', linewidth=1.5, markersize=5)
  plt.errorbar(number_dimension_list, avg_irm_T_pehe, std_irm_T_pehe, color='orange', marker='>', linestyle='dashed', linewidth=1.5, markersize=5)
  plt.errorbar(number_dimension_list, avg_erm_S_pehe, std_erm_S_pehe, color='green', marker='v', linestyle='dotted', linewidth=1.5, markersize=5)
  plt.errorbar(number_dimension_list, avg_erm_T_pehe, std_erm_T_pehe, color='red', marker='^', linestyle='dashed', linewidth=1.5, markersize=5)
  plt.rc('legend', fontsize = 18)
  plt.xticks([10,20,30,40,50], fontsize = 15)
  if args.feature_model == "A" and args.outcome_model == "linear":
    plt.yticks([5,10,15,20,25],fontsize = 15)
  elif args.feature_model == "B" and args.outcome_model == "linear":
    plt.yticks([5,10,15,20,25],fontsize = 15)
  elif args.feature_model == "A" and args.outcome_model == "quadratic":
    plt.yticks([300,600,900,1200,1500],fontsize = 15)
  else:
    plt.yticks([300,600,900,1200,1500],fontsize = 15)
  plt.rc('figure', titlesize = 30)
  matplotlib.rcParams['text.usetex'] = True
  plt.xlabel(r'$d$',fontsize = 18)
  plt.ylabel(r'$\sqrt{\epsilon_{PEHE}}$',fontsize = 20)
  plt.legend(handles=[darkblue, orange, green, red], loc='upper left')
  plt.tight_layout()
  plt.savefig('feature_model_'+str(args.feature_model)+'_outcome_model_'+str(args.outcome_model)+'_mu_'+str(args.mu)+'_pehe.eps')

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
  
  args = parser.parse_args() 
  
  # Calls main function  
  main(args)