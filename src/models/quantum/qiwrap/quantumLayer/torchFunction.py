# Author : Maniraman Periyasamy
# This code is part of guided-spsa repository.
# This code uses parts of code snippets from qiskit
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.



import torch
from torch.autograd import Function
import numpy as np


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, QcN, input, weights, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = QcN
        expectation_z = ctx.quantum_circuit.execute_circuit(input, weights)
        result = torch.FloatTensor(expectation_z)
        ctx.save_for_backward(input, weights, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):

        input, weights, expectation_z = ctx.saved_tensors
        grad = ctx.quantum_circuit.execute_circuit(
            input, weights, calc_gradient=True)
        batch_size = len(grad)
        if len(grad.shape) == 2:
            weights_grad = torch.einsum(
                "ki,ij->j", grad_output.detach().cpu(), torch.FloatTensor(grad))
        else:
            weights_grad = torch.einsum(
                "ij,ijk->k", grad_output.detach().cpu(), torch.FloatTensor(grad))
        weights_grad = weights_grad.to(weights.device)

        return None, None, weights_grad, None
