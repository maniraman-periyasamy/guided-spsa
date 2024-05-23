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




from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.result.sampled_expval import sampled_expectation_value
from qiskit.quantum_info import Pauli
from qiskit.result import QuasiDistribution
from qiskit.providers.jobstatus import JobStatus

# Type setting imports
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit import Parameter as qiskitParameter
#from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
import torch

import numpy as np
import multiprocessing as mp
import copy
import sys
import time
import datetime

from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence
)



def calculate_forward(
                        executer,
                        circuit,
                        ind,
                        params,
                        mp_result,
                        observables=None,
                        shots=None,
                        ret=False):
    
    job = executer.run(
            circuits=circuit, observables=observables, parameter_values=params, shots=shots)
    #ceck = job.result().values
    result = np.array(job.result().values).flatten()

    if ret:
        return result

    mp_result[ind[0]:ind[-1]+1] = result


def calculate_grads(Grad_estimator, circuit, ind, params, n_thetas,
                    parameter_set, mp_result, observables=None,shots=None, ret=False):

    job = Grad_estimator.run(
        circuits=circuit, observables=observables,
        parameter_values=params,
        parameters=parameter_set, shots=shots)
    result = np.array(job.result().gradients).flatten()

    if ret:
        return result

    mp_result[ind[0]*n_thetas:ind[0]*n_thetas+len(result)] = result


class base_executer:
    """This class implements the forward probagation and gradient estimation step for different executers.

        Args:
            qc (quantumcircuit): Variational Quantum ansatz
            weight_params (Union[ ParameterVector, Sequence[qiskitParameter]]): List of weight parameters in the VQC
            input_params (Union[ParameterVector, list[qiskitParameter]]): List of input parameters in the VQC
            batch_size (int): Batch size for parallelization and SPSA
            observables (Optional[Sequence[BaseOperator  |  PauliSumOp]], optional): List of quantum observables. Defaults to None.
            grad_type ({"SPSA, "Parameter-shift"}, optional): Gradient estimation method to be used. Defaults to "SPSA".
            epsilon (Optional[float], optional): Epsilon for SPSA. This will be ignored in case of parameter-shift type gradient estimation Defaults to 0.2.
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
            shots (Optional[int], optional): Number of shots to be used for shot based simulator. Defaults to 1024.
            ibmq_executer (Optional[bool], optional): Flag to indicate ibmq executer device. Defaults to False.
            ibmq_session_max_time (Optional[int], optional): Maximum time for ibmq session in seconds. Defaults to 25200

        """
    def __init__(
                self,
                qc: QuantumCircuit,
                weight_params: Union[
                                    ParameterVector,
                                    Sequence[qiskitParameter]],
                input_params: Union[ParameterVector, list[qiskitParameter]],
                batch_size: int,
                observables: Optional[Sequence[BaseOperator]] = None,
                grad_type: Optional[str] = "SPSA",
                epsilon: Optional[float] = 0.2,
                g_spsa_param_ratio: Optional[float] = 0.5,
                num_parallel_process: Optional[int] = None,
                shots: Optional[int] = None,
                grad_reset: Optional[int] = 10,
                grad_decay: Optional[float] = 0.1,
                grad_norm_length: Optional[int] = 1,
                ibmq_executer: Optional[bool] = False,
                ibmq_session_max_time: Optional[int] = 7*60*60) -> None:

        self.qcm = qc
        self.weight_parameters = weight_params
        self.encoding_parmeters = input_params
        self.observables = observables
        self.batch_size = batch_size
        self.grad_type = grad_type
        self.epsilon = epsilon
        self.num_parallel_process = num_parallel_process
        self.shots = shots
        self.executer = None
        self.Grad_estimator = None
        self.Grad_executer_spsa = None
        self.Grad_executer_param = None
        self.grad_reset = grad_reset
        self.grad_decay = grad_decay
        self.grad_norm_length = grad_norm_length
        self.weight_grads = []
        self.prev_epoch_grad = None
        self.prev_grad_ref = None

        self.grad_reset_counter = 0
        self.grad_decay_epsilon = 1.0
        self.grad_norm_counter = 0
        self.grad_norm = np.zeros((self.grad_norm_length))
        self.grad_norm_flag = False
        self.spsa_grad_angle = []
        self.g_spsa_grad_angle = []
        self.ps_grad_norm = []
        self.spsa_grad_norm = []
        self.g_spsa_grad_norm = []
        self.renorm_g_spsa_grad_norm = []

        self.ibmq_executer = ibmq_executer
        self.ibmq_session_max_time = ibmq_session_max_time

        self.parameter_shift_counter = 0
        self.spsa_counter = 0

        self.ps_ratio = g_spsa_param_ratio
        self.spsa_damping = 0.5
        self.spsa_start_idx = None

        if self.observables is None:
            if len(self.qcm.clbits) == 0:
                self.qcm.measure_all()
            self.num_outputs = 2**len(self.qcm.clbits)
        else:
            self.num_outputs = len(self.observables)

        self.ibmq_session_timeup_flag = False
        self.ibmq_session_init_time = datetime.datetime.now()
        self.ibqm_session_init_time_set_flag = False
        self.duration = 0

    def set_executer(self, session=None, options=None, noise_model=None, coupling_map=None, basis_gates=None):
        raise NotImplementedError

    
    def split_indices(self, params):
        if self.num_parallel_process is None:
            num_process = mp.cpu_count()
        else:
            num_process = self.num_parallel_process
        
        indices = []
        start = 0
        size = len(params) // num_process
        remainder = len(params)%num_process

        for i in range(num_process):
            if i < remainder:
                end = start+size+1
            else:
                end = start+size
            indices += [list(range(start, end))]
            start = end

        

        indices += [list(range(end, len(params)))]

        indices = [x for x in indices if x != []]
        return copy.deepcopy(indices)
    
    
    def execute_circuit(
                self, inputs: Union[np.ndarray, Sequence[float]],
                thetas: Union[np.ndarray, Sequence[float]],
                calc_gradient: Optional[bool] = False):
        
        """This fuction executes the given VQC.

        Args:
            inputs (Union[np.ndarray, Sequence[float]]): List of inputs to the VQC
            thetas (Union[np.ndarray, Sequence[float]]): List of weights for the VQC
            calc_gradient (Optional[bool], optional): Flag to indicate gradient estimation. Defaults to False.

        Returns:
            np.ndarray: VQC execution results in case 'calc_gradient' is false or gradients.
        """

        thetas_br = np.broadcast_to(thetas, (len(inputs), len(thetas)))
        params = np.concatenate((thetas_br, inputs), axis=1)

        
        if len(inputs) != 1 :# and not self.ibmq_executer:
            #if False:
            if not calc_gradient:
                # if False:
                params = np.tile(params, (len(self.observables), 1))
                op = [o for o in self.observables for _ in range(len(inputs))]
                indices = self.split_indices(params)
                result = mp.Array('f', len(params))
                process = [
                        mp.Process(
                            target=calculate_forward,
                            args=(
                                self.executer, [self.qcm]*len(inds), inds,
                                params[inds], result, [copy.deepcopy(op[i])for i in inds], self.shots,
                                ))
                        for inds in indices]
                
                for p in process:
                    p.start()

                for p in process:
                    p.join()
                
                result_np = np.array(result, dtype=np.float32)
                result_np = np.reshape(result_np, (-1, len(inputs))).T

                return result_np

            else:
                
                self.spsa_start_idx = int(len(inputs)*self.ps_ratio)
                params_ps = params[:self.spsa_start_idx]
                params_spsa = params[self.spsa_start_idx:]

                # parameter_shift
                params_ps = np.tile(params_ps, (len(self.observables), 1))
                op_ps = [o for o in self.observables for _ in range(self.spsa_start_idx)]
                indices_ps = self.split_indices(params_ps)
                result_ps = mp.Array('f', (len(params_ps) * len(thetas)))
                process_param = [
                        mp.Process(
                            target=calculate_grads,
                            args=(
                                self.Grad_executer_param, [self.qcm]*len(inds),
                                inds, params_ps[inds], len(thetas),
                                [self.weight_parameters.params]*len(inds),
                                result_ps, [copy.deepcopy(op_ps[i])for i in inds], self.shots))
                        for inds in indices_ps]

                # spsa
                params_spsa = np.tile(params_spsa, (len(self.observables), 1))
                op_spsa = [o for o in self.observables for _ in range(len(inputs)-self.spsa_start_idx)]
                indices_spsa = self.split_indices(params_spsa)
                result_spsa = mp.Array('f', (len(params_spsa) * len(thetas)))
                process_spsa = [
                        mp.Process(
                            target=calculate_grads,
                            args=(
                                self.Grad_executer_spsa, [self.qcm]*len(inds),
                                inds, params_spsa[inds], len(thetas),
                                [self.weight_parameters.params]*len(inds),
                                result_spsa, [copy.deepcopy(op_spsa[i])for i in inds], self.shots))
                        for inds in indices_spsa]
                process = process_param + process_spsa

                for p in process:
                    p.start()

                for p in process:
                    p.join()

                result_np_ps = np.array(result_ps, dtype=np.float32).reshape(len(params_ps), len(thetas))
                result_np_spsa =np.array(result_spsa, dtype=np.float32).reshape(len(params_spsa), len(thetas))
                weights_grad = np.zeros((len(inputs), len(self.observables), len(thetas)))
                for i in range(len(self.observables)):
                    weights_grad[:self.spsa_start_idx, i, :] = result_np_ps[i * self.spsa_start_idx: (i + 1) * self.spsa_start_idx]
                    weights_grad[self.spsa_start_idx:, i, :] = result_np_spsa[i * (len(inputs)-self.spsa_start_idx): (i + 1) * (len(inputs)-self.spsa_start_idx)]
                del result_ps, result_spsa
                
                return weights_grad
        else:
            params = np.tile(params, (len(self.observables), 1))
            op = [o for o in self.observables for _ in range(len(inputs))]
            if not calc_gradient:
                result = None
                result_np = calculate_forward(
                        self.executer, [self.qcm] * len(params),
                        list(range(len(params))), params, result,
                        observables=op, shots=self.shots, ret=True)

                result_np = result_np.reshape(len(inputs), self.num_outputs)
                
                return result_np
            else:
                result = None
                
                ps_index = int(len(inputs)*self.ps_ratio)
                
                self.spsa_start_idx = ps_index
                params_ps = params[:self.spsa_start_idx]
                prams_g_spsa = params[self.spsa_start_idx:]
                result_ps= calculate_grads(
                        self.Grad_executer_param, [self.qcm]*len(params_ps),
                        list(range(len(params_ps))), params_ps, len(thetas),
                        [self.weight_parameters.params]*len(params_ps),
                        result, op[:self.spsa_start_idx], shots=self.shots, ret=True)

                result_g_spsa= calculate_grads(
                        self.Grad_executer_spsa, [self.qcm]*len(prams_g_spsa),
                        list(range(len(prams_g_spsa))), prams_g_spsa, len(thetas),
                        [self.weight_parameters.params]*len(prams_g_spsa),
                        result, op[self.spsa_start_idx:], shots=self.shots, ret=True)
                result = np.concatenate([result_ps, result_g_spsa])
                result_np = np.array(result, dtype=np.float32).reshape(len(params), len(thetas))
                weights_grad = np.zeros((len(inputs), self.num_outputs, len(thetas)))
                for i in range(len(self.observables)):
                    weights_grad[:, i, :] = result_np[i * len(inputs): (i+1)*len(inputs)]
                
                
                return weights_grad
                
