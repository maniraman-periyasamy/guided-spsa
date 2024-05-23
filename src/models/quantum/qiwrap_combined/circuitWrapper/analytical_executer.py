# Author: Maniraman Periyasamy (maniraman.periyasamy@iis.fraunhofer.de)

from qiskit_algorithms.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_aer.primitives import Estimator
from qiskit_aer.primitives import Sampler


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
from ...qiwrap_combined.circuitWrapper.base_executer import base_executer
#from ...qiwrap_combined.customGradiets.spsa_estimator_gradient import SPSAEstimatorGradient
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
            grad_type: Optional[str] = "SPSA", epsilon: Optional[float] = 0.2, 
            g_spsa_param_ratio: Optional[float] = 0.5,
            num_parallel_process: Optional[int] = None,
            grad_reset: Optional[int] = None,
                grad_decay: Optional[float] = None) -> None:

        super(analytical_executer, self).__init__(
            qc=qc, weight_params=weight_params, input_params=input_params, batch_size=batch_size,
            observables=observables, grad_type=grad_type, epsilon=epsilon, g_spsa_param_ratio=g_spsa_param_ratio, num_parallel_process=num_parallel_process, grad_decay=grad_decay, grad_reset=grad_reset)

    def set_executer(self, session=None, options=None, noise_model=None, coupling_map=None, basis_gates=None, grads = None):
        """This function sets the executer for the vqc and the respective gradient estimation method.

        Args:
            session (QiskitSession, optional): This argument is ignoerd for analytical estimation. Defaults to None.
        """

        if self.grad_type == "SPSA":
            self.executer = Estimator(approximation=True)
            self.Grad_executer = SPSAEstimatorGradient(self.executer, epsilon=self.epsilon, batch_size=self.batch_size)
        elif self.grad_type == "Guided-SPSA":
            self.executer = Estimator()
            self.Grad_executer_spsa = SPSAEstimatorGradient(self.executer, epsilon=self.epsilon, batch_size=self.batch_size)
            self.Grad_executer_param = ParamShiftEstimatorGradient(self.executer)
        else:
            self.executer = Estimator(approximation=True)
            self.Grad_executer = ParamShiftEstimatorGradient(self.executer)
