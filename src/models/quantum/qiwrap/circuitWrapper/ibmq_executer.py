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


from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_ibm_runtime import Sampler, Estimator, Options
# Type setting imports
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit import Parameter as qiskitParameter
#from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from typing import (
    Any,
    Dict,
    Union,
    Optional,
    Sequence
)
import datetime

# Custom modules
from ...qiwrap.circuitWrapper.base_executer import base_executer


class ibmq_executer(base_executer):

    """This class implements the executer for IBMQ devices.

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
            ibmq_session_max_time (Optional[int], optional): Maximum time for ibmq session in seconds. Defaults to 25200

    """
    def __init__(
            self,  qc: QuantumCircuit, weight_params: Union[ParameterVector, Sequence[qiskitParameter]], input_params: Union[ParameterVector, list[qiskitParameter]],
            batch_size: int, observables: Optional[Sequence[BaseOperator]] = None,
            grad_type: Optional[str] = "SPSA", epsilon: Optional[float] = 0.2, num_parallel_process: Optional[int] = None, shots: Optional[int] = None,
            ibmq_session_max_time: Optional[int] = 7*60*60) -> None:

        super(ibmq_executer, self).__init__(
            qc=qc, weight_params=weight_params, input_params=input_params, batch_size=batch_size,
            observables=observables, grad_type=grad_type, epsilon=epsilon, num_parallel_process=num_parallel_process, shots=shots, ibmq_executer=True,
            ibmq_session_max_time=ibmq_session_max_time)

    def set_executer(self, session=None, options=None, noise_model=None, coupling_map=None, basis_gates=None):
        """This function sets the executer for the vqc and the respective gradient estimation method.

        Args:
            session (QiskitRuntime Session, optional): Qiskit runtime session to use. Defaults to None.
            options (QiskitRuntime Session Options, optional): Transpilation and error mitigation options. If None, highest possible setting is used. Defaults to None.
        """
        self.ibmq_session_timeup_flag = False
        self.ibmq_session_init_time = datetime.datetime.now()
        self.ibqm_session_init_time_set_flag = False
        
        if options is None:
            Options(optimization_level=3, resilience_level=2, simulator={'noise_model':noise_model, 'coupling_map': coupling_map, 'basis_gates': basis_gates})

        if self.grad_type == "SPSA":
            self.executer = Estimator(session=session, options=options)
            self.Grad_executer = SPSAEstimatorGradient(self.executer, epsilon=self.epsilon, batch_size=self.batch_size)
        else:
            self.executer = Estimator(session=session, options=options)
            self.Grad_executer = ParamShiftEstimatorGradient(self.executer)
