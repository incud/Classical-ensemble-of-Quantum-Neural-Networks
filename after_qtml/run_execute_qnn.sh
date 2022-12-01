echo "Running execution..."

python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e01_seed1000 --mode jax --varform hardware_efficient --layers 1 --seed 9000