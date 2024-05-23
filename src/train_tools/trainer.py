from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import sklearn.metrics as met

from train_tools.logger import ExperimentTracker, Stage
from train_tools.metric import Metric

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Estimator, Options

class executer:
    def __init__(self, data_loader: DataLoader[Any], net: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
    sk_metric = None, type: str = "regression", classification_type: str = "unique", num_classes: int = 4, executer_type = "simulator", grad_type = "SPSA") -> None:

        self.run_count = 0
        self.data_loader = data_loader
        self.accuracy_metric = Metric()
        self.loss_metric = Metric()
        self.net = net
        self.optimizer = optimizer
        self.sk_metric = sk_metric
        # Objective (loss) function
        if type == "regression":
            self.compute_loss = torch.nn.MSELoss()
        if type == "classification":
            self.compute_loss = torch.nn.BCELoss()
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        # Assume Stage based on presence of optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN
        self.type = type
        self.classification_type = classification_type
        self.num_classes = num_classes
        self.onehot = OneHotEncoder(sparse_output=False, categories=list(range(num_classes)))
        self.executer_type = executer_type
        self.grad_type = grad_type
        self.grad_angle: list[list[Any]] = []
        self.grad_angle_mean: float = 0
        self.grad_circuit_counter = 0
        self.batch_counter = 0
        self.ps_exted_counter = 9
        self.curr_train_loss = np.array([np.inf]*5)
        self.ps_prioritize = True
        

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average
    
    @property
    def avg_loss(self):
        return self.loss_metric.average

    def run(self, desc: str, experiment: ExperimentTracker, epoch_id: int, best_epoch: int, options=None, service=None):
        self.net.train(self.stage is Stage.TRAIN)
        epoch_toggle = True
        for iter, (x, y) in enumerate(tqdm(self.data_loader, desc=desc, ncols=80)):
            
            #if self.grad_type=="Guided-SPSA" and  epoch_id - best_epoch == 1 and iter<self.ps_exted_counter :
            #if self.grad_type=="Guided-SPSA" and  self.ps_prioritize:# and iter<self.ps_exted_counter :
            #    #self.net.QcN.weight_grads = None
            #    self.net.QcN.grad_reset_counter = 0
            #    self.net.QcN.grad_decay_epsilon = 1.0
            


            batch_flag = True
            session_created = False
            while batch_flag:
                loss, batch_accuracy = self._run_single(x, y, self.stage is Stage.TRAIN)
                try:
                    loss, batch_accuracy = self._run_single(x, y, self.stage is Stage.TRAIN)
                    batch_flag = False
                except:
                    print("Batch_failed re-running!!!")
                    
            experiment.add_batch_metric("accuracy", batch_accuracy, self.run_count)
            experiment.add_batch_loss("loss", loss.item(), self.run_count)

            if self.optimizer:
                
                # Reverse-mode AutoDiff (backpropagation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iter == 0 and self.grad_type=="Guided-SPSA":
                    self.param_shift_samples = self.net.QcN.spsa_start_idx
                
                self.batch_counter += 1
                if self.grad_type=="Param-shift":
                    self.grad_circuit_counter+= len(x)*len(self.net.trainable_parameters)*2*len(self.net.QcN.observables)
                elif self.grad_type=="SPSA":
                    self.grad_circuit_counter+= len(x)*self.net.QcN.Grad_executer._batch_size*2*len(self.net.QcN.observables)


                if self.grad_type=="Guided-SPSA":
                    self.grad_circuit_counter+= (len(x[self.net.QcN.spsa_start_idx:]))*self.net.QcN.Grad_executer_spsa._batch_size*2*len(self.net.QcN.observables)
                    self.grad_circuit_counter+= (len(x[:self.net.QcN.spsa_start_idx]))*len(self.net.trainable_parameters)*2*len(self.net.QcN.observables)

                
       
        if self.optimizer and epoch_id < 20:
            experiment.add_batch_histogram("gradients", self.net.trainable_parameters.grad, epoch_id)

                    

    def _run_single(self, x: Any, y: Any, train_flag:bool = False):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.net(x)
        if self.type == "classification":
            prediction = torch.nn.Softmax()(prediction)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Validation Metrics
        y_np = y.detach().numpy()
        y_prediction_np = prediction.detach().numpy()
        if self.type == "regression":
            batch_accuracy: float = self.sk_metric(y_np, y_prediction_np)
        elif self.type == "classification":
            if not train_flag:
                y_class = torch.argmax(y, axis=-1)
                prediction_class = torch.argmax(prediction, axis=-1)
                accuracy = - torch.sum(y_class == prediction_class)/len(y_class)
                batch_accuracy: float = accuracy.item()#self.sk_metric(y_np, y_prediction_oneHot)
            else:
                batch_accuracy: float = loss.item()
        self.accuracy_metric.update(batch_accuracy, batch_size)
        self.loss_metric.update(loss.item(), batch_size)

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]
        return loss, batch_accuracy

    def reset(self):
        self.accuracy_metric = Metric()
        self.loss_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []


def execute_epoch(
    train_runner: executer,
    validation_runner: executer,
    test_runner: executer,
    experiment: ExperimentTracker,
    epoch_id: int,
    best_epoch: int,
    options=None, 
    service=None
):
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("Train Batches", experiment, epoch_id, best_epoch, options=options, service=service)

    # Log Training Epoch Metrics
    experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_loss("loss", train_runner.avg_loss, epoch_id)

    # Testing Loop
    experiment.set_stage(Stage.VAL)
    validation_runner.run("Validation Batches", experiment, epoch_id, best_epoch, options=options, service=service)
    
    experiment.set_stage(Stage.TEST)
    test_runner.run("Test Batches", experiment, epoch_id, best_epoch, options=options, service=service)

    # Log Validation Epoch Metrics
    experiment.add_epoch_metric("accuracy", validation_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_loss("loss", validation_runner.avg_loss, epoch_id)
    