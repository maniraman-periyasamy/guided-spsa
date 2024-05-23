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





from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli

from ..quantum.qiwrap.quantumLayer import torch_layer
from ..quantum.qiwrap_combined.quantumLayer import torch_layer as combined_torch_layer
import os
import numpy as np

def skolik_arch(n_qubits, n_layers, batch_size, grad_type = "SPSA", spsa_epsilon=0.45, 
quantum_compute_method="analytical", spsa_batch_size=2, n_features=4, repeat=1, quantum_weight_initialization="random",
g_spsa_param_ratio = 0.5, observables = None):

    circuit = QuantumCircuit(n_qubits)

    
    inputParams = ParameterVector("x", length=n_features)
    circuitParams = ParameterVector("psi", length=2*n_layers*n_qubits)

    data_counter = 0
    # Construct the variational layers
    for i in range(n_layers):
        for j in range(n_qubits):
            if data_counter != n_features:
               circuit.rx(inputParams[data_counter], j)
               data_counter += 1 
            circuit.ry(circuitParams[2*i*n_qubits + j], j)
            circuit.rz(circuitParams[(2*i+1)*n_qubits + j], j)
        
        for j in range(n_qubits-1):
            circuit.cz(j, (j+1))
        circuit.cz(n_qubits-1, 0)
        
        circuit.barrier()
    
    if repeat != None and quantum_weight_initialization == "random":
        
        if os.path.isfile("src/models/quantum/weights/"+str(repeat%5)+"_"+str(2*n_layers*n_qubits)+".npy"):
            quantum_weights = np.load("src/models/quantum/weights/"+str(repeat%5)+"_"+str(2*n_layers*n_qubits)+".npy")
        else:
            if not os.path.exists("src/models/quantum/weights/"):
                os.makedirs("src/models/quantum/weights/")
            quantum_weights = np.random.uniform(low=0.0, high=np.pi, size=2*n_layers*n_qubits).astype(np.float32)
            np.save("src/models/quantum/weights/"+str(repeat%5)+"_"+str(2*n_layers*n_qubits)+".npy", quantum_weights)
    else:
        quantum_weights = None
    if type(observables) == type(None):
        observables = [SparsePauliOp(Pauli("Z"*n_qubits))]
    if grad_type == "Guided-SPSA":
        net = combined_torch_layer(qc=circuit, weight_params=circuitParams, input_params=inputParams, observables=observables, batch_size=spsa_batch_size, grad_type=grad_type,
                epsilon=spsa_epsilon, output_scaling=True, # data_reuploading_layers=n_layers, input_scaling=False, add_input_weights=True, add_output_weights=True, data_reuploading_layers=n_layers,
                quantum_weight_initialization=quantum_weight_initialization, quantum_weights=quantum_weights,
                num_parallel_process=None, shots=1024, quantum_compute_method=quantum_compute_method,
                ibmq_session_max_time=7200, grad_decay=0.4, grad_reset=5, g_spsa_param_ratio=g_spsa_param_ratio)
    else:
        net = torch_layer(qc=circuit, weight_params=circuitParams, input_params=inputParams, observables=observables, batch_size=spsa_batch_size, grad_type=grad_type,
                epsilon=spsa_epsilon, output_scaling=True,#, input_scaling=False, add_input_weights=True, add_output_weights=True, data_reuploading_layers=n_layers, 
                quantum_weight_initialization=quantum_weight_initialization, quantum_weights=quantum_weights,
                num_parallel_process=None, shots=1024, quantum_compute_method=quantum_compute_method,
                ibmq_session_max_time=7200)
    return net