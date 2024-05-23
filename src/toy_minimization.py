import torch

import numpy as np
import math
import os
from os import environ
environ['OPENBLAS_NUM_THREADS'] = '1'


import hydra
from hydra.core.config_store import ConfigStore
from config import regressionConfig
from optuna.trial import Trial
from omegaconf import OmegaConf



from models.classical.model import Net
from models.quantum.model import skolik_arch
from qiskit_aer.noise import NoiseModel


def loss_fn(outputs):
    l = torch.sin(outputs/2) + torch.sin(2.3*torch.sin(4*outputs))
    l = torch.mean(l)
    return l



cs = ConfigStore.instance()
cs.store(name="regression_config", node=regressionConfig)
@hydra.main(config_path="conf", config_name="config_reg", version_base="1.2")
def main(cfg: regressionConfig) -> float:
    test_avg = 0.0
    converegence_avg = 0.0
    datasets = [ "Any"]
    for dataset_name in datasets:
        rep_avg = 0
        converegence_rep_avg = 0
        for rep in range(cfg.algorithm_params.repeats):
            dataset_name = cfg.dataset_params.name
            if dataset_name == "friedman1":
                cfg.quantum_model_params.n_qubits = 5
            else:
                cfg.quantum_model_params.n_qubits = 4
           
            if dataset_name == "friedman1":
                cfg.dataset_params.n_features = 5
            elif dataset_name == "boston-housing":
                cfg.dataset_params.n_features = 13
            
            if cfg.algorithm_params.method == "classical":
                net = Net(
                    n_feature=cfg.dataset_params.n_features, 
                    n_hidden_layer=cfg.classical_model_params.hidden_layers, 
                    n_hidden_units=cfg.classical_model_params.hidden_units,
                    n_output=1
                    )
                print(net)
            else:
                noisy_flag = False
                if cfg.algorithm_params.computation_type == "noisy":
                    cfg.algorithm_params.computation_type = "shot-based"
                    noisy_flag = True
                net=skolik_arch(n_qubits=cfg.quantum_model_params.n_qubits, 
                        n_layers=cfg.quantum_model_params.layers,
                        batch_size=cfg.dataset_params.batch_size, grad_type=cfg.algorithm_params.gradient_type, 
                        spsa_epsilon=cfg.algorithm_params.spsa_epsilon, 
                        quantum_compute_method=cfg.algorithm_params.computation_type, spsa_batch_size=cfg.algorithm_params.spsa_batch_size,
                        n_features= cfg.dataset_params.n_features, repeat=rep, quantum_weight_initialization=cfg.quantum_model_params.quantum_weight_initialization,
                        g_spsa_param_ratio = cfg.algorithm_params.g_spsa_param_ratio)
                net.load_state_dict(torch.load('toy_problem/best_model.pth'))
                if noisy_flag:
                    from qiskit_ibm_provider import IBMProvider
                    # Save your credentials on disk.
                    IBMProvider.save_account("ADD API Token",
                        overwrite=True)
                    provider = IBMProvider(instance='ibm-q/open/main')
                    backend = provider.get_backend('ibm_brisbane')
                    noise_model = NoiseModel.from_backend(backend)
                    coupling_map = backend.coupling_map
                    basis_gates = noise_model.basis_gates
                    net.set_session(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
                else:
                    net.set_session()

            optim = torch.optim.Adam(net.parameters(), lr=0.1)
            
            log_root = cfg.log_params.tb_log + cfg.algorithm_params.method + "_" + cfg.algorithm_params.gradient_type + "_" + \
                cfg.algorithm_params.optimizer+"_"+str(cfg.algorithm_params.spsa_batch_size)+"_/"+ dataset_name+ "_"+ str(rep)+"/" +str(cfg.algorithm_params.lr)+"_"+str(cfg.algorithm_params.spsa_epsilon)+"/"
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            log_root = log_root + hydra_cfg['runtime']['output_dir'].split("/guided-spsa/",1)[1]
            
            if not os.path.exists(log_root):
                os.makedirs(log_root)
            # dumps to file:
            with open(log_root+"/config.yaml", "w") as f:
                OmegaConf.save(cfg, f)

            spsa_end_batch_ratio = 1 - cfg.algorithm_params.g_spsa_param_ratio + 0.5
            grad_batch_size_start = max(1, math.ceil(len(net.trainable_parameters)/10))
            
            for epoch_id in range(cfg.algorithm_params.epochs):
                if cfg.algorithm_params.gradient_type == "Guided-SPSA":
                    grad_batch_size_val = max(1,math.floor(grad_batch_size_start + epoch_id*(( len(net.trainable_parameters)*spsa_end_batch_ratio - grad_batch_size_start)/cfg.algorithm_params.epochs)))
                    net.QcN.Grad_executer_spsa._batch_size = grad_batch_size_val
                    net.QcN.spsa_damping = cfg.algorithm_params.spsa_damping_factor
                optim.zero_grad()
                input = np.zeros((32, 4))
                output = net(input)
                loss = loss_fn(output)
                loss.backward()
                optim.step()

                summary = ", ".join(
                        [  
                            f"Epoch: {epoch_id + 1}/{cfg.algorithm_params.epochs}]",
                            f"x: {np.mean(output.detach().numpy()): 0.4f}",
                            f"y: {loss.item(): 0.4f}"
                        ]
                    )
                print("\n" + summary + "\n")
            output = net(np.zeros((1,4)))
            loss = loss_fn(output)
            print(f"Test Metric: {loss.item(): 0.4f}")

    
    return None


if __name__ == "__main__":

    main()