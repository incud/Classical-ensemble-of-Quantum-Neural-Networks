echo "Running execution..."

for VARFORM in hardware_efficient tfim ltfim
do
  for LAYER in 1 5 10
  echo "Running execution for varform $VARFORM and layers $LAYER"
  do
    python execute_qnn.py experiment --dataset datasets/linear/n250_d02_e01_seed1000 --mode jax --varform $VARFORM --layers $LAYER --seed 9000
  done
done