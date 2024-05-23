# Guide-SPSA

This repository provides the experimental code for work "[Guided-SPSA: Simultaneous Perturbation Stochastic Approximation assisted by the Parameter Shift Rule by M. Periyasamy et. al.](https://arxiv.org/abs/2404.15751)". However, this repo is an initial experimental implementation of guided-spsa. For the installable module of the guided-spsa gradient evaluation, please refer to the following git repository "[gspsa-gradients](https://github.com/maniraman-periyasamy/gspsa-gradients)".


## Setup and Installation

We recommend setting up a conda environment and install the required python packages using the ``environment.yml`` file:

```bash
conda env create -f environment.yml
```

## Running the experiments

We use Hyrda to configure the experimental hyperparameters and run the respective experiments.

### Regression

The default hyperparameters for the regression experiments are in the file "config_reg.yaml" under the directory "src/conf". One can adjust the hyperparameters and run the experiment as shown below:

```bash
python -u src/regression.py hydra/job_logging=disabled
```

All parameters in "config_reg.yaml" can also be adjusted during the Python execution as shown below:

```bash
python -u src/regression.py hydra/job_logging=disabled 'algorith_params.lr=0.03'
```


### Classification

The default hyperparameters for the regression experiments are in the file "config_clas.yaml" under the directory "src/conf". One can adjust the hyperparameters and run the experiment as shown below:

```bash
python -u src/classification.py hydra/job_logging=disabled
```

All parameters in "config_clas.yaml" can also be adjusted during the Python execution as shown below:

```bash
python -u src/classification.py hydra/job_logging=disabled 'algorith_params.lr=0.03'
```

## Acknowledgements

We use ``qiskit`` software framework: https://github.com/Qiskit


## Citation

If you use the `gspsa-gradients` or results from the paper, please cite our work as

```
@misc{periyasamy2024guidedspsa,
      title={Guided-SPSA: Simultaneous Perturbation Stochastic Approximation assisted by the Parameter Shift Rule}, 
      author={Maniraman Periyasamy and Axel Plinge and Christopher Mutschler and Daniel D. Scherer and Wolfgang Mauerer},
      year={2024},
      eprint={2404.15751},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## License

Apache 2.0 License
