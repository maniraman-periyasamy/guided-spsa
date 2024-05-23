# Author : Maniraman Periyasamy
# This code is part of guided-spsa repository.
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.



from dataclasses import dataclass

@dataclass
class Dataset_Params:
  dataset_size: int
  n_features: int
  batch_size: int
  validation_size: float
  test_size: float
  name: str
  noise: float
  num_classes: int


@dataclass
class Classical_model_params:
  hidden_units: int
  hidden_layers: int

@dataclass
class Quantum_model_params:
  layers: int
  scaling_weights_lr: float
  n_qubits: int
  quantum_weight_initialization: str


@dataclass
class Algorithm_Params:
  method: str
  lr: float
  optimizer: str
  epochs: int
  gradient_type: str
  computation_type: str
  spsa_epsilon: float
  spsa_batch_size: int
  g_spsa_param_ratio: float
  spsa_damping_factor: float
  opt_momentum: float
  opt_rho: float
  repeats: int

@dataclass
class Log_Params:
  tb_log: str
  model_log: str
  matplot_log: str
  test_res_log: str


@dataclass
class regressionConfig:
    dataset_params: Dataset_Params
    algorithm_params: Algorithm_Params
    log_params: Log_Params
    classical_model_params: Classical_model_params
    quantum_model_params: Quantum_model_params