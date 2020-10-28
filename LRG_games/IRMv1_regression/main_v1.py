
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sem import ChainEquationModel
from models_v1 import *

import argparse
import torch
import numpy

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
    train_size = np.int(args["n_samples"]*0.8) ## split data into 80:20 train:val
    val_size = np.int(args["n_samples"]*0.2)
    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = ChainEquationModel(args["dim"],
                                     hidden=args["setup_hidden"],
                                     scramble=args["setup_scramble"],
                                     hetero=args["setup_hetero"])

            env_list = args["env_list"]
            m = len(env_list)
            environments = []
            for o in range(2*m):
                if(o<m):
                    environments.append(sem(train_size, env_list[o%m]))
                    environments.append(sem(train_size, env_list[o%m]))
                else:
                    environments.append(sem(val_size, env_list[o%m]))
                    environments.append(sem(val_size, env_list[o%m]))                  
                
#             environments = [sem(args["n_samples"], e) for e in env_list]
            
#             environments = [sem(args_ns1, 0.2),
#                             sem(args_ns1, 2.0),
#             sem(args_ns2, 0.2),
#             sem(args_ns2, 2.0)                
#             ]
                        
#             environments = [sem(train_size, 0.2),
#                             sem(train_size, 2.0),
#             sem(val_size, 0.2),
#             sem(val_size, 2.0)                
#             ]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    for sem, environments in zip(all_sems, all_environments):
        sem_solution, sem_scramble = sem.solution()
        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                             pretty(sem_solution), 0, 0)
        ]
        

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)
            msolution = sem_scramble @ method.solution()
            err_causal, err_noncausal = errors(sem_solution, msolution)

            solutions.append("{} {} {} {:.5f} {:.5f}".format(setup_str,
                                                             method_name,
                                                             pretty(msolution),
                                                             err_causal,
                                                             err_noncausal))

        all_solutions += solutions

    return all_solutions, all_environments, msolution, sem_solution


def run_experiment(args):
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
                                     hetero=args["setup_hetero"])
            environments = [sem(args["n_samples"], 0.2),
                            sem(args["n_samples"], 2.0)
            ]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)
        
    for sem, environments in zip(all_sems, all_environments):
        sem_solution, sem_scramble = sem.solution()
        solutions = [
            "{} SEM {} {:.5f} {:.5f}".format(setup_str,
                                             pretty(sem_solution), 0, 0)
        ]
        

        for method_name, method_constructor in methods.items():
            method = method_constructor(environments, args)
            msolution = sem_scramble @ method.solution()
            err_causal, err_noncausal = errors(sem_solution, msolution)

            solutions.append("{} {} {} {:.5f} {:.5f}".format(setup_str,
                                                             method_name,
                                                             pretty(msolution),
                                                             err_causal,
                                                             err_noncausal))

        all_solutions += solutions


    return all_solutions, all_environments, msolution, sem_solution


