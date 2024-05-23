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

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_layer,n_hidden_units, n_output):
        super(Net, self).__init__()
        
        layers = [torch.nn.Linear(n_feature, n_hidden_units)]
        layers.append(torch.nn.ReLU())
        for i in range(0, n_hidden_layer-1):
            layers.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
        self.predict = torch.nn.Linear(n_hidden_units, n_output)
        
    def forward(self, x):

        x = self.layers(x)
        x = self.predict(x)             # linear output
        return x