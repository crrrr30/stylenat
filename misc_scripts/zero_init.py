# From https://arxiv.org/pdf/2110.12661.pdf

import torch
import math
from scipy.linalg import hadamard

def ZerO_Init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper.
    
    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)
    
    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
    
    return init_matrix

def Identity_Init_on_matrix(matrix_tensor):
    # Definition 1 in the paper
    # See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.eye_ for details. Preserves the identity of the inputs in Linear layers, where as many inputs are preserved as possible, the same as partial identity matrix.
    
    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)
    
    init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    
    return init_matrix


# def _init_weights(self, m):
#     if isinstance(m, nn.Linear):
#         m.weight.data = ZerO_Init_on_matrix(m.weight.data)