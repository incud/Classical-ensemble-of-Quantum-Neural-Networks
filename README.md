# Classical-ensemble-of-Quantum-Neural-Networks

This repository contains the code to reproduce the experiment in [arXiv:2303.11283](https://arxiv.org/abs/2303.11283).

## How to reproduce the experiments

Commands to run specific configurations:

    python execute_qnn_concrete.py experiment --dataset datasets/concrete --dataset-type concrete --mode jax --varform hardware_efficient --layers 1 --seed 1000

    python execute_qnn_diabete.py experiment --dataset datasets/diabete --dataset-type diabete --mode jax --varform hardware_efficient --layers 1 --seed 1000

    python execute_qnn.py experiment --dataset datasets/linear/n250_d05_e01_seed1001 --dataset-type linear --mode jax --varform hardware_efficient --layers 1 --seed 1000

    python execute_qnn_noise.py experiment --dataset datasets/linear/n250_d05_e01_seed1001 --dataset-type linear --mode jax --varform hardware_efficient --layers 1 --seed 1000

Commands to run ALL simulations of a given dataset:

    sh run_execute_qnn_concrete.sh 

    sh run_execute_qnn_diabete.sh 

    sh run_execute_qnn.sh 

## How to cite

    @article{incudini2023resource,
        title={Resource saving via ensemble techniques for quantum neural networks},
        author={Incudini, Massimiliano and Grossi, Michele and Ceschini, Andrea and Mandarino, Antonio and Panella, Massimo and Vallecorsa, Sofia and Windridge, David},
        journal={arXiv preprint arXiv:2303.11283},
        year={2023}
    }
