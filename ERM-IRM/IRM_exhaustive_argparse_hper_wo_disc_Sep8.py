
### This file is based on https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/experiment_synthetic/models.py
### This is the main file which calls the ERM and IRM models and generates the final comparison plots

from datetime import date
import time
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pandas as pd
tf.compat.v1.enable_eager_execution()
import cProfile
from sklearn.model_selection import train_test_split
import copy as cp
from sklearn.model_selection import KFold

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sem_Sep8 import ChainEquationModel
from models_crossval_Sep8 import *
import argparse
import torch
import numpy

print ("Cross val IRM")
t_start = time.time()


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = (w != 0).nonzero().view(-1)
    i_noncausal = (w == 0).nonzero().view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    return error_causal, error_noncausal


def run_experiment_ERM(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "chain_hidden={}_hetero={}_scramble={}".format(
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_solutions = []
    all_environments = []

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(args["dim"],
                                     hidden=args["setup_hidden"],
                                     scramble=args["setup_scramble"],
                                     hetero=args["setup_hetero"], child = args["child"],  noise_identity = args["noise_identity"], ones=args["ones"])
            environments = [sem(args["n_samples"], 0.2),
            sem(args["n_samples"], 2.0)
            ]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    for sem, environments in zip(all_sems, all_environments):
        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                             pretty(sem.solution()), 0, 0)
        ]

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)
            msolution = method.solution()
            sem_solution = sem.solution()
            method_sol_out = msolution
            err_causal, err_noncausal = errors(sem.solution(), msolution)

            solutions.append("{} {} {} {:.5f} {:.5f}".format(setup_str,
                                                             method_name,
                                                             pretty(msolution),
                                                             err_causal,
                                                             err_noncausal))

        all_solutions += solutions

    return all_solutions, all_environments, msolution, sem_solution



def run_experiment_IRM(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "chain_hidden={}_hetero={}_scramble={}".format(
            args["setup_hidden"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "ICP": InvariantCausalPrediction,
        "IRM": InvariantRiskMinimization
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_solutions = []
    all_environments = []
    frac = 0.8 
    args_ns = args["n_samples"]
    args_ns1 = np.int(args_ns * frac)
    args_ns2 = args_ns-args_ns1
    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(args["dim"],
                                     hidden=args["setup_hidden"],
                                     scramble=args["setup_scramble"],
                                     hetero=args["setup_hetero"], child = args["child"],  noise_identity = args["noise_identity"], ones=args["ones"])
            environments = [sem(args_ns1, 0.2),
                            sem(args_ns1, 2.0),
            sem(args_ns2, 0.2),
            sem(args_ns2, 2.0) 
            ]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    for sem, environments in zip(all_sems, all_environments):
        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                             pretty(sem.solution()), 0, 0)
        ]

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)
            msolution = method.solution()
            sem_solution = sem.solution()
            method_sol_out = msolution
            err_causal, err_noncausal = errors(sem.solution(), msolution)

            solutions.append("{} {} {} {:.5f} {:.5f}".format(setup_str,
                                                             method_name,
                                                             pretty(msolution),
                                                             err_causal,
                                                             err_noncausal))

        all_solutions += solutions

    return all_solutions, all_environments, msolution, sem_solution



    


def run_irm_exhaustive(args):


    result = pd.DataFrame(columns = ["dim", "scramble", "hetero", "hidden", "child", "noise_identity","ones", "sample", "method", "error", "error_std"])


    n_repits = args["n_repits"]
    scramble = args["setup_scramble"]
    hetero   = args["setup_hetero"]
    hidden   = args["setup_hidden"]
    child    = args["child"]
    ones     = args["ones"]
    noise_identity = args["noise_identity"]

    # sample_list = [50, 100, 500,  1000, 1500, 2000]
    sample_list = [50, 200, 500,  1000, 1500, 2000]
    c=0


    lambd = 0.9
    n_s = len(sample_list)
    Error_ERM_f_cont = np.zeros(n_s)
    Error_IRMv1_f_cont = np.zeros(n_s)
    k=0
    list_params= []
    for n_sample in sample_list:
        args["n_samples"] = n_sample
        Error_ERM_cont = np.zeros(n_repits)
        Error_IRMv1_cont = np.zeros(n_repits)

        for repits in range(n_repits):     
            dim =4
            args["seed"] = repits


            args["methods"] = "IRM"
            all_solutions, all_environments, msolution, sem_solution = run_experiment_IRM(args)
            
            print("\n".join(all_solutions))
            true_sol = sem_solution.detach().numpy().T[0]
            IRMv1_cont = msolution.detach().numpy().T[0]
            msolution.detach().numpy().T[0]
            
            Error_IRMv1_cont[repits] = np.linalg.norm(IRMv1_cont-true_sol)**2

            args["methods"] = "ERM"
            all_solutions, all_environments, msolution, sem_solution = run_experiment_ERM(args)
            
            print("\n".join(all_solutions))
            true_sol = sem_solution.detach().numpy().T[0]
            ERM_cont = msolution.detach().numpy()[0]
            Error_ERM_cont[repits] = np.linalg.norm(ERM_cont-true_sol)**2



        list_params.append([dim, scramble, hetero, hidden, child, noise_identity, ones, n_sample, "ERM_cont", np.mean(Error_ERM_cont),np.std(Error_ERM_cont)])
        list_params.append([dim, scramble, hetero, hidden, child, noise_identity, ones, n_sample, "IRMv1_cont", np.mean(Error_IRMv1_cont),np.std(Error_IRMv1_cont)])

       
        Error_ERM_f_cont[k] = np.mean(Error_ERM_cont)
        Error_IRMv1_f_cont[k] = np.mean(Error_IRMv1_cont)

        


        k=k+1

    plt.figure() 
    plt.xlabel("Number of samples", fontsize=16)
    plt.ylabel("Model estimation error", fontsize=16)
    plt.plot(sample_list, Error_ERM_f_cont, "-r", marker="+", label="ERM")
    plt.plot(sample_list, Error_IRMv1_f_cont, "-b", marker="s", label="IRMv1")
    plt.legend(loc="upper left", fontsize=18)

    dir_name = "------enter path to the directory-----"
    fil_name = str(date.today())[5:10] + "_cval_" +"_hid_" + str(args["setup_hidden"]) + "_n_repits_" + str(args["n_repits"]) + "_dim_" + str(args["dim"]) + "_scramble_" + str(args["setup_scramble"]) + "_hetero_"  + str(args["setup_hetero"]) + "_child_" + str(args["child"])  + "_noise_" +str(args["noise_identity"]) + "_ones_" + str(args["ones"]) 
    plt.savefig(dir_name + fil_name + ".pdf")






    result = pd.DataFrame(list_params, columns = ["dim", "scramble", "hetero", "hidden","child", "noise_identity","ones", "sample", "method", "error", "error_std"])
    result.to_csv(dir_name + fil_name + ".csv")





parser = argparse.ArgumentParser(description='Sample complexity invariant regression')
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--n_reps', type=int, default=1)
parser.add_argument('--skip_reps', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)  # Negative is random
parser.add_argument('--print_vectors', type=int, default=1)
parser.add_argument('--n_iterations', type=int, default=50000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.05)
parser.add_argument('--setup_sem', type=str, default="chain")
parser.add_argument('--setup_hidden', type=int, default=0)
parser.add_argument('--setup_hetero', type=int, default=0)
parser.add_argument('--setup_scramble', type=int, default=0)
parser.add_argument('--noise_identity', type=int, default=0)
parser.add_argument('--ones', type=int, default=0)
parser.add_argument('--child', type=int, default=0)
parser.add_argument('--n_repits', type=int, default=1)
args_dict = dict(vars(parser.parse_args()))
print (args_dict)

run_irm_exhaustive(args_dict)
t_end = time.time()
print ("total time "  + str(t_end-t_start))

