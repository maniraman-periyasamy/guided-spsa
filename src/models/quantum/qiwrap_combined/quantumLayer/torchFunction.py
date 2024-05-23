import torch
from torch.autograd import Function
import numpy as np
import math
import sys


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


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
        grad_norm = np.linalg.norm(grad, ord=2, axis=2)
        sp_start = ctx.quantum_circuit.spsa_start_idx
        spsa_norm_suppresent = np.mean(grad_norm[:sp_start], axis=0)/grad_norm[sp_start:]
        
        spsa_norm_suppresent = spsa_norm_suppresent*ctx.quantum_circuit.spsa_damping
        mult_factor = np.where(spsa_norm_suppresent<=1.0, spsa_norm_suppresent, 1.0)
        

        grad[sp_start:] = np.multiply(grad[sp_start:], mult_factor[:, :, np.newaxis])
        batch_size = len(grad)
        if len(grad.shape) == 2:
            weights_grad = torch.einsum(
                "ki,ij->j", grad_output.detach().cpu(), torch.FloatTensor(grad))
        else:
            weights_grad = torch.einsum(
                "ij,ijk->k", grad_output.detach().cpu(), torch.FloatTensor(grad))
        weights_grad = weights_grad.to(weights.device)
        
            
        return None, None, weights_grad, None
