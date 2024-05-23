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
        self.compute_loss = torch.nn.MSELoss()
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
        self.ps_exted_counter = 5
        self.curr_train_loss = -np.inf
        self.ps_prioritize = False

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
            if self.grad_type=="Guided-SPSA" and  self.ps_prioritize:# and iter<self.ps_exted_counter :
                self.net.QcN.weight_grads = None
                self.net.QcN.grad_reset_counter = 0


            if self.net.check_time_up_flag():
                if self.executer_type == "simulator":
                    session = Session(service=service, backend="ibmq_qasm_simulator", max_time=14400)
                else:
                    session = Session(service=service, backend="ibmq_ehiningen", max_time=14400)
                self.net.set_session(session=session, options=options)
            batch_flag = True
            session_created = False
            while batch_flag:
                try:
                    if session_created:
                        print("new session running")
                    loss, batch_accuracy = self._run_single(x, y)
                    batch_flag = False
                except:
                    print("Batch_failed re-running!!!")
                    if self.executer_type == "simulator":
                        session = Session(service=service, backend="ibmq_qasm_simulator", max_time=14400)
                    else:
                        #pass
                        print("New sesion starting")
                        session = Session(service=service, backend="ibmq_ehiningen", max_time=14400)
                        session_created = True
                        print("neqw session created")
                    self.net.set_session(session=session, options=options)

            experiment.add_batch_metric("accuracy", batch_accuracy, self.run_count)
            experiment.add_batch_loss("loss", loss.item(), self.run_count)

            if self.optimizer:
                
                # Reverse-mode AutoDiff (backpropagation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                
                self.batch_counter += 1
                if self.grad_type=="Param-shift":
                    self.grad_circuit_counter+= len(x)*len(self.net.trainable_parameters)*2
                elif self.grad_type=="SPSA":
                    self.grad_circuit_counter+= len(x)*self.net.QcN.Grad_executer._batch_size*2


                if self.grad_type=="Guided-SPSA" and type(self.net.QcN.weight_grads) != type(None):
                    #experiment.add_batch_angle("Grad-angle", self.net.QcN.grad_angle[-1], self.run_count)
                    #self.grad_angle.append(self.net.QcN.grad_angle[-1])
                    self.grad_circuit_counter+= len(x)*self.net.QcN.Grad_executer_spsa._batch_size*2
                if self.grad_type=="Guided-SPSA" and type(self.net.QcN.weight_grads) == type(None):
                    self.net.QcN.weight_grads = self.net.trainable_parameters.grad.clone().detach().numpy()
                    self.grad_circuit_counter+= len(x)*len(self.net.trainable_parameters)*2
                    #self.grad_angle = 0

                
        if self.optimizer and epoch_id < 20:
            experiment.add_batch_histogram("gradients", self.net.trainable_parameters.grad, epoch_id)
            #self.grad_angle_mean = np.mean(self.grad_angle)
            #self.grad_angle = []
        if self.optimizer:
            if self.accuracy_metric.average >= self.curr_train_loss:
                #print("red")
                self.ps_prioritize = True
                self.curr_train_loss = self.accuracy_metric.average
            else:
                self.ps_prioritize = False
                self.curr_train_loss = self.accuracy_metric.average

    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.net(x)
        loss = self.compute_loss(prediction, y)

        # Compute Batch Validation Metrics
        y_np = y.detach().numpy()
        if self.type == "regression":
            y_prediction_np = prediction.detach().numpy()
        else:
            if self.classification_type == "unique":
                y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)

                y_prediction_np = self.onehot.fit_transform(y_prediction_np)
            else:
                y_prediction_np = prediction.detach().numpy()
                y_prediction_np = (y_prediction_np < 0.5)*y_prediction_np
        batch_accuracy: float = self.sk_metric(y_np, y_prediction_np)
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
    