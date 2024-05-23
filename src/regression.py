import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


import matplotlib.pyplot as plt
import numpy as np
import math
import os
from sklearn.metrics import mean_absolute_error
from os import environ
environ['OPENBLAS_NUM_THREADS'] = '1'


import hydra
from hydra.core.config_store import ConfigStore
from config import regressionConfig
from optuna.trial import Trial
from omegaconf import DictConfig
from omegaconf import OmegaConf



from models.classical.model import Net
from models.quantum.model import skolik_arch
from data.dataset_generator import generate_dataloader
from train_tools.trainer import executer, execute_epoch
from train_tools.logger import TensorboardLogger, Stage

from train_tools.save_best_model import SaveBestModel
from qiskit_aer.noise import NoiseModel




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
                elif cfg.algorithm_params.computation_type == "error-mitig":
                    cfg.algorithm_params.computation_type = "ibmq"
                    noisy_flag = True
                net=skolik_arch(n_qubits=cfg.quantum_model_params.n_qubits, 
                        n_layers=cfg.quantum_model_params.layers,
                        batch_size=cfg.dataset_params.batch_size, grad_type=cfg.algorithm_params.gradient_type, 
                        spsa_epsilon=cfg.algorithm_params.spsa_epsilon, 
                        quantum_compute_method=cfg.algorithm_params.computation_type, spsa_batch_size=cfg.algorithm_params.spsa_batch_size,
                        n_features= cfg.dataset_params.n_features, repeat=rep, quantum_weight_initialization=cfg.quantum_model_params.quantum_weight_initialization,
                        g_spsa_param_ratio = cfg.algorithm_params.g_spsa_param_ratio)
                if noisy_flag:
                    from qiskit_ibm_provider import IBMProvider
                    # Save your credentials on disk.
                    IBMProvider.save_account("ADD API TOKEN",
                        overwrite=True)
                    provider = IBMProvider(instance='ibm-q/open/main')
                    backend = provider.get_backend('ibm_brisbane')
                    noise_model = NoiseModel.from_backend(backend)
                    coupling_map = backend.coupling_map
                    basis_gates = noise_model.basis_gates
                    if cfg.algorithm_params.computation_type == "shot-based":
                        net.set_session(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
                    elif cfg.algorithm_params.computation_type == "ibmq":
                        from qiskit_ibm_runtime import QiskitRuntimeService, Session

                        service = QiskitRuntimeService(
                            channel='ibm_quantum',
                            instance='ibm-q/open/main',
                            token='ADD API TOKEN'
                        )
                        session = Session(service=service, backend="ibmq_qasm_simulator")
                        net.set_session(session=session, noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
                else:
                    net.set_session()

            if cfg.algorithm_params.optimizer == "Adam":
                if cfg.algorithm_params.method == "classical":
                    optim = torch.optim.Adam(net.parameters(), lr=cfg.algorithm_params.lr, betas=(cfg.algorithm_params.opt_momentum,cfg.algorithm_params.opt_rho))
                else:
                    optim = torch.optim.Adam([{'params': net.trainable_parameters},
                        {'params': net.output_scaling_layer.parameters(), 'lr': cfg.quantum_model_params.scaling_weights_lr}], 
                        lr=cfg.algorithm_params.lr, betas=(cfg.algorithm_params.opt_momentum,cfg.algorithm_params.opt_rho))
            
            
            train_loader, test_loader, validation_loader = generate_dataloader(
                task="regression",
                name=cfg.dataset_params.name, 
                dataset_size=cfg.dataset_params.dataset_size, 
                n_features=cfg.dataset_params.n_features, 
                noise=cfg.dataset_params.noise, 
                test_size=cfg.dataset_params.test_size, 
                validation_size=cfg.dataset_params.validation_size, 
                batch_size=cfg.dataset_params.batch_size)

            train_executer = executer(train_loader, net=net, optimizer=optim, sk_metric=mean_absolute_error, type="regression", grad_type=cfg.algorithm_params.gradient_type)
            validation_executer = executer(validation_loader, net=net, sk_metric=mean_absolute_error, type="regression", grad_type=cfg.algorithm_params.gradient_type)
            test_executer = executer(test_loader, net=net, sk_metric=mean_absolute_error, type="regression", grad_type=cfg.algorithm_params.gradient_type)

            log_root = cfg.log_params.tb_log + cfg.algorithm_params.method + "_" + cfg.algorithm_params.gradient_type + "_" + \
                cfg.algorithm_params.optimizer+"_"+str(cfg.algorithm_params.spsa_batch_size)+"_/"+ dataset_name+ "_"+ str(rep)+"/" +str(cfg.algorithm_params.lr)+"_"+str(cfg.algorithm_params.spsa_epsilon)+"/"
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            log_root = log_root + hydra_cfg['runtime']['output_dir'].split("/guided-spsa/",1)[1]
            log_obj = TensorboardLogger(log_path=log_root)
            model_saver = SaveBestModel(location=log_root)
            
            # dumps to file:
            with open(log_root+"/config.yaml", "w") as f:
                OmegaConf.save(cfg, f)

            spsa_end_batch_ratio = 1 - cfg.algorithm_params.g_spsa_param_ratio + 0.5
            grad_batch_size_start = max(1, math.ceil(len(net.trainable_parameters)/10))
            
            for epoch_id in range(cfg.algorithm_params.epochs):
                if cfg.algorithm_params.gradient_type == "Guided-SPSA":
                    grad_batch_size_val = max(1,math.floor(grad_batch_size_start + epoch_id*(( len(net.trainable_parameters)*spsa_end_batch_ratio - grad_batch_size_start)/cfg.algorithm_params.epochs)))
                    train_executer.net.QcN.Grad_executer_spsa._batch_size = grad_batch_size_val
                    train_executer.net.QcN.spsa_damping = cfg.algorithm_params.spsa_damping_factor
                    
                execute_epoch(train_runner=train_executer, validation_runner=validation_executer, test_runner=test_executer, epoch_id=epoch_id, best_epoch=model_saver.best_epoch, experiment=log_obj)
                if cfg.algorithm_params.gradient_type == "Guided-SPSA":
                    # Compute Average Epoch Metrics
                    summary = ", ".join(
                        [  
                            f"[Dataset: {dataset_name}, Epoch: {epoch_id + 1}/{cfg.algorithm_params.epochs}]",
                            f"Train Metric: {train_executer.avg_accuracy: 0.4f}",
                            f"Val Metric: {validation_executer.avg_accuracy: 0.4f}",
                            f"TM: {test_executer.avg_accuracy: 0.4f}",
                            f"Grad_circuits: {train_executer.grad_circuit_counter}",
                            f"Prameter-samples: {train_executer.param_shift_samples}"

                        ]
                    )
                    
                else:
                    summary = ", ".join(
                        [  
                            f"[Dataset: {dataset_name}, Epoch: {epoch_id + 1}/{cfg.algorithm_params.epochs}]",
                            f"Train Metric: {train_executer.avg_accuracy: 0.4f}",
                            f"Val Metric: {validation_executer.avg_accuracy: 0.4f}",
                            f"TM: {test_executer.avg_accuracy: 0.4f}",
                            f"Grad_circuits: {train_executer.grad_circuit_counter}"
                        ]
                    )
                print("\n" + summary + "\n")
                model_saver(current_valid_loss=validation_executer.avg_accuracy, epoch=epoch_id, model=net, 
                    param_ct=net.QcN.parameter_shift_counter, spsa_ct=net.QcN.spsa_counter)

                # Reset the runners
                train_executer.reset()
                test_executer.reset()
                validation_executer.reset()

                log_obj.flush()
            grad_histogram = np.array(log_obj.grads)
            np.save(log_root+"/grad_historgram.npy", grad_histogram)
            net.load_state_dict(torch.load(log_root+"/best_model.pth"))
            log_obj.set_stage(Stage.TEST)
            test_executer.run("Test", log_obj, epoch_id=epoch_id, best_epoch=model_saver.best_epoch)
            print(f"Test Metric: {test_executer.avg_accuracy: 0.4f}, Convergence Epoch: {model_saver.best_epoch}, param-ct: {model_saver.param_ct}, spsa-ct: {model_saver.spsa_ct}, param-ct_ov: {net.QcN.parameter_shift_counter}, spsa-ct_ov: {net.QcN.spsa_counter}, total circs: {train_executer.grad_circuit_counter}")
            f = open(log_root + "/test_metric.txt", "a")
            f.write(f"Test Metric: {test_executer.avg_accuracy}")
            f.close()
            #plt.scatter(np.arange(len(test_executer.y_true_batches)), test_executer.y_true_batches, c="red")
            #plt.scatter(np.arange(len(test_executer.y_pred_batches)), test_executer.y_pred_batches, c="green")
            #plt.savefig(log_root + "/"+cfg.dataset_params.name +"_test.png")
            rep_avg = rep_avg + test_executer.avg_accuracy
            converegence_rep_avg += model_saver.best_epoch
        test_avg += rep_avg/cfg.algorithm_params.repeats
        converegence_avg += converegence_rep_avg/cfg.algorithm_params.repeats 
    print(f"{log_root} yieled an acccuracy of {test_avg/len(datasets)} and convergence rate of {converegence_avg/len(datasets)}")
    return test_avg/len(datasets)

def configure(cfg: regressionConfig, trial: Trial) -> None:

    trial.suggest_loguniform("algorithm_params.lr", 0.001, 0.1)  # note +w here, not w as w is a new parameter
    trial.suggest_uniform("algorithm_params.spsa_epsilon", 0.1, 0.6)

    if cfg.algorithm_params.optimizer == "SGD" or cfg.algorithm_params.optimizer == "Pure_SPSA":  # "Adam", "SGD", "AMSGrad", "RMSProp"
        trial.suggest_loguniform("algorithm_params.opt_momentum", 0.001, 0.2)
        trial.suggest_loguniform("algorithm_params.opt_rho", 0.001, 0.2)
        
    elif cfg.algorithm_params.optimizer == "Adam" or cfg.algorithm_params.optimizer == "AMSGrad":
        trial.suggest_uniform("algorithm_params.opt_momentum", 0.1, 0.9)
        trial.suggest_uniform("algorithm_params.opt_rho", 0.1, 0.999)

    elif cfg.algorithm_params.optimizer == "RMSProp":
        trial.suggest_loguniform("algorithm_params.opt_momentum", 0.001, 0.2)
        trial.suggest_uniform("algorithm_params.opt_rho", 0.1, 0.999)

if __name__ == "__main__":

    main()