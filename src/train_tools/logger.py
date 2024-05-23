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




from enum import Enum, auto
from pathlib import Path
from typing import Protocol

import numpy as np
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class ExperimentTracker(Protocol):
    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""
    
    def add_batch_loss(self, name: str, value: float, step: int):
        """Implements logging a batch-level loss."""

    def add_epoch_loss(self, name: str, value: float, step: int):
        """Implements logging a epoch-level loss."""
    def add_batch_angle(self, name: str, value: float, step: int):
        """Implements logging a epoch-level loss."""
    def add_batch_histogram(self, name: str, value: float, step: int):
        """Implements logging a epoch-level loss."""


class TensorboardLogger:
    def __init__(self, log_path: str, create: bool = True):
        self.stage = Stage.TRAIN
        self._writer = SummaryWriter(log_dir=log_path)
        self.grads = []
        self.bins = np.arange(-0.15, 0.155, 0.01)
        plt.ioff()

    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        self._writer.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_batch_loss(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_loss(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_batch_angle(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_batch_histogram(self, name: str, value: torch.tensor, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_histogram(tag, value, step)
        self.grads.append(value.clone().detach().numpy())