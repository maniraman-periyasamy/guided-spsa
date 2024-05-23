import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


import matplotlib.pyplot as plt
import numpy as np
import math
import os
from sklearn.metrics import mean_absolute_error
from os import environ
#environ['OPENBLAS_NUM_THREADS'] = '1'



from models.classical.model import Net
from models.quantum.model import skolik_arch
from data.dataset_generator import generate_dataloader
from train_tools.trainer import executer, execute_epoch
from train_tools.logger import TensorboardLogger, Stage

from train_tools.save_best_model import SaveBestModel
from qiskit_aer.noise import NoiseModel



net = skolik_arch(n_qubits=4, n_features=4, n_layers=5, batch_size=1, grad_type="Param-shift")
optim = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()
net.set_session()
for i in range(200):
    optim.zero_grad()
    input = np.zeros((32, 4))
    output = net(input)
    loss = loss_fn(torch.ones(32)*(np.pi/2), output)
    loss.backward()
    optim.step()
    print(i, np.mean(output.detach().numpy()), loss.item())

st  = net.state_dict()
if not os.path.exists("toy_problem/"):
   os.makedirs("toy_problem/")
torch.save(st, 'toy_problem/best_model.pth')

np.save('toy_problem/quantum_weights.npy', st['trainable_parameters'].detach().numpy())
np.save('toy_problem/scaling_weights.npy', st['output_scaling_layer.trainable_parameters'].detach().numpy())