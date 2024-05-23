# Author: Maniraman Periyasamy (maniraman.periyasamy@iis.fraunhofer.de)


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
                num_parallel_process: Optional[int] = None,
                shots: Optional[int] = None,
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
        self.ibmq_executer = ibmq_executer
        self.ibmq_session_max_time = ibmq_session_max_time
        self.parameter_shift_counter = 0
        self.spsa_counter = 0

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

        if self.observables is not None:
            params = np.tile(params, (len(self.observables), 1))
            op = [o for o in self.observables for _ in range(len(inputs))]

        if len(inputs) != 1 and not self.ibmq_executer:
            # if False:
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

            if not calc_gradient:
                # if False:
                result = mp.Array('f', len(params))
                process = [
                        mp.Process(
                            target=calculate_forward,
                            args=(
                                self.executer, [self.qcm]*len(inds), inds,
                                params[inds], result, [op[i]for i in inds], self.shots,
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
                result = mp.Array('f', (len(params) * len(thetas)))
                process = [
                        mp.Process(
                            target=calculate_grads,
                            args=(
                                self.Grad_executer, [self.qcm]*len(inds),
                                inds, params[inds], len(thetas),
                                [self.weight_parameters.params]*len(inds),
                                result, [op[i]for i in inds], self.shots))
                        for inds in indices]

                for p in process:
                    p.start()

                for p in process:
                    p.join()

                result_np = np.array(result, dtype=np.float32).reshape(len(params), len(thetas))
                weights_grad = np.zeros((len(inputs), len(self.observables), len(thetas)))
                for i in range(len(self.observables)):
                    weights_grad[:, i, :] = result_np[i * len(inputs): (i + 1) * len(inputs)]
                del result
                return weights_grad
        else:
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

                result = calculate_grads(
                        self.Grad_executer, [self.qcm]*len(params),
                        list(range(len(params))), params, len(thetas),
                        [self.weight_parameters.params]*len(params),
                        result, op, shots=self.shots, ret=True)

                result_np = np.array(result, dtype=np.float32).reshape(len(params), len(thetas))
                weights_grad = np.zeros((len(inputs), self.num_outputs, len(thetas)))
                for i in range(len(self.observables)):
                    weights_grad[:, i, :] = result_np[i * len(inputs): (i+1)*len(inputs)]
                return weights_grad
