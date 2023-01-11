echo "Running execution..."

for VARFORM in hardware_efficient
do
  for LAYER in 1 2 3 4 5 6 7 8 9 10
  do
    echo "Running execution for varform $VARFORM and layers $LAYER"
    python execute_qnn.py experiment --dataset datasets/linear/n250_d05_e01_seed1001 --dataset-type linear --mode jax --varform $VARFORM --layers $LAYER --seed 1000
  done
done