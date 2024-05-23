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




import torch


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf'), location='outputs/'
    ):
        self.best_valid_loss = best_valid_loss
        self.location = location
        self.best_epoch = -1
        self.param_ct = 0
        self.spsa_ct = 0
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, param_ct, spsa_ct
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_epoch = epoch
            self.param_ct = param_ct
            self.spsa_ct = spsa_ct
            #print(f"\nBest validation loss: {self.best_valid_loss}")
            #print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), self.location+'/best_model.pth')