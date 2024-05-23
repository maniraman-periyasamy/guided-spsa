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



import torch

import numpy as np
import os


from models.quantum.model import skolik_arch


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