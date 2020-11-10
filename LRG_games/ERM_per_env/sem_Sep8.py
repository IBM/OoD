

### This file is based on (slight modification of) https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/experiment_synthetic/sem.py
### We included the functionality to control whether to allow for anti-causal variables or not

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

class ChainEquationModel(object):
    def __init__(self, dim, scramble=False, hetero=True, hidden=False, child=True,  ones=True, noise_identity=True):
        self.hetero = hetero
        self.hidden = hidden
        self.dim = dim // 2
        print ("ones" + str(ones))
        

        if ones:
            self.wxy = torch.eye(self.dim)
            if child:
                print ("child " + str(child))
                self.wyz = torch.eye(self.dim)
            else:
                self.wyz =torch.zeros(self.dim, self.dim)
        else:
            self.wxy = torch.randn(self.dim, self.dim) / dim
            if child:
                self.wyz = torch.randn(self.dim, self.dim) / dim
            else:
                self.wyz =torch.zeros(self.dim, self.dim)
        if scramble:
            self.scramble, _ = torch.qr(torch.randn(dim, dim))
        else:
            self.scramble = torch.eye(dim)

        if hidden:
            if noise_identity==0:
                print ("noise_identity " + str(noise_identity))
                self.whx = torch.randn(self.dim, self.dim) / dim 
                self.why = torch.randn(self.dim, self.dim) / dim 
                self.whz = torch.randn(self.dim, self.dim) / dim 
            else:
                if noise_identity==1:
                    print ("noise_identity " + str(noise_identity))
                    self.whx = torch.eye(self.dim, self.dim)
                    self.why = torch.eye(self.dim, self.dim)
                    self.whz = torch.eye(self.dim, self.dim)
                else:
                    if noise_identity==2:
                        print ("noise_identity " + str(noise_identity))
                        self.whx = torch.rand(self.dim, self.dim) / dim
                        self.why = torch.rand(self.dim, self.dim) / dim
                        self.whz = torch.rand(self.dim, self.dim) / dim                    

        else:
            self.whx = torch.eye(self.dim, self.dim) 
            self.why = torch.zeros(self.dim, self.dim)
            self.whz = torch.zeros(self.dim, self.dim)


    def solution(self):
        w = torch.cat((self.wxy.sum(1), torch.zeros(self.dim))).view(-1, 1)
        return self.scramble.t() @ w

    def __call__(self, n, env):
        h = torch.randn(n, self.dim) * env
#         x = h @ self.whx + torch.randn(n, self.dim) * env
        x = torch.randn(n, self.dim) * env
        if self.hetero:
            print ("hetero " + str(self.hetero))
            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim) * env
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim)
        else:

            y = x @ self.wxy + h @ self.why + torch.randn(n, self.dim)
            z = y @ self.wyz + h @ self.whz + torch.randn(n, self.dim) * env

        return torch.cat((x, z), 1) @ self.scramble, y.sum(1, keepdim=True)


