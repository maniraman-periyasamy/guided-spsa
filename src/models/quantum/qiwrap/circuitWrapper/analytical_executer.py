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
from qiskit_aer.primitives import Estimator, Sampler


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

# Custom modules
from ...qiwrap.circuitWrapper.base_executer import base_executer
from sklearn.preprocessing import normalize


class analytical_executer(base_executer):
    """This class implements the analytical executer for a given VQC.

        Args:
            qc (quantumcircuit): Variational Quantum ansatz
            weight_params (Union[ ParameterVector, Sequence[qiskitParameter]]): List of weight parameters in the VQC
            input_params (Union[ParameterVector, list[qiskitParameter]]): List of input parameters in the VQC
            batch_size (int): Batch size for parallelization and SPSA
            observables (Optional[Sequence[BaseOperator  |  PauliSumOp]], optional): List of quantum observables. Defaults to None.
            grad_type ({"SPSA, "Parameter-shift"}, optional): Gradient estimation method to be used. Defaults to "SPSA".
            epsilon (Optional[float], optional): Epsilon for SPSA. This will be ignored in case of parameter-shift type gradient estimation Defaults to 0.2.
            num_parallel_process (Optional[int], optional): Number of parallel process generated. If None, number of parallel process equals the number of available cores. Defaults to None.
    """

    def __init__(
            self,  qc: QuantumCircuit, weight_params: Union[ParameterVector, Sequence[qiskitParameter]], input_params: Union[ParameterVector, list[qiskitParameter]],
            batch_size: int, observables: Optional[Sequence[BaseOperator]] = None,
            grad_type: Optional[str] = "SPSA", epsilon: Optional[float] = 0.2, num_parallel_process: Optional[int] = None) -> None:

        super(analytical_executer, self).__init__(
            qc=qc, weight_params=weight_params, input_params=input_params, batch_size=batch_size,
            observables=observables, grad_type=grad_type, epsilon=epsilon, num_parallel_process=num_parallel_process)

    def set_executer(self, session=None, options=None, noise_model=None, coupling_map=None, basis_gates=None):
        """This function sets the executer for the vqc and the respective gradient estimation method.

        Args:
            session (QiskitSession, optional): This argument is ignoerd for analytical estimation. Defaults to None.
        """

        if self.grad_type == "SPSA":
            self.executer = Estimator()
            self.Grad_executer = SPSAEstimatorGradient(self.executer, epsilon=self.epsilon, batch_size=self.batch_size)
        else:
            self.executer = Estimator()
            self.Grad_executer = ParamShiftEstimatorGradient(self.executer)
