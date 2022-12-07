echo "Running execution..."

for VARFORM in hardware_efficient tfim ltfim
do
  for LAYER in 1 5 10
  echo "Running execution for varform $VARFORM and layers $LAYER"
  do
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e01_seed1000 --mode jax --varform $VARFORM --layers $LAYER --seed 9000
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e01_seed1000 --mode jax --varform $VARFORM --layers $LAYER --seed 9001
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e01_seed2000 --mode jax --varform $VARFORM --layers $LAYER --seed 9002
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e01_seed3000 --mode jax --varform $VARFORM --layers $LAYER --seed 9003
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e05_seed1010 --mode jax --varform $VARFORM --layers $LAYER --seed 9004
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e05_seed2010 --mode jax --varform $VARFORM --layers $LAYER --seed 9005
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d02_e05_seed3010 --mode jax --varform $VARFORM --layers $LAYER --seed 9006
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d05_e01_seed1001 --mode jax --varform $VARFORM --layers $LAYER --seed 9007
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d05_e01_seed2002 --mode jax --varform $VARFORM --layers $LAYER --seed 9008
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d05_e01_seed3003 --mode jax --varform $VARFORM --layers $LAYER --seed 9009
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d05_e05_seed1011 --mode jax --varform $VARFORM --layers $LAYER --seed 9010
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d05_e05_seed2012 --mode jax --varform $VARFORM --layers $LAYER --seed 9011
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d05_e05_seed3013 --mode jax --varform $VARFORM --layers $LAYER --seed 9012
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d10_e01_seed1004 --mode jax --varform $VARFORM --layers $LAYER --seed 9013
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d10_e01_seed2005 --mode jax --varform $VARFORM --layers $LAYER --seed 9014
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d10_e01_seed3006 --mode jax --varform $VARFORM --layers $LAYER --seed 9015
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d10_e05_seed1014 --mode jax --varform $VARFORM --layers $LAYER --seed 9016
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d10_e05_seed2015 --mode jax --varform $VARFORM --layers $LAYER --seed 9017
    python3.9 execute_qnn.py experiment --dataset datasets/n250_d10_e05_seed3016 --mode jax --varform $VARFORM --layers $LAYER --seed 9018
  done
done