echo "Running execution..."

for VARFORM in hardware_efficient
do
  for LAYER in 1 2 3 4 5 6 7 8 9 10
  do
    echo "Running execution for varform $VARFORM and layers $LAYER"
    python execute_qnn_concrete.py experiment --dataset datasets/concrete --dataset-type concrete --mode jax --varform $VARFORM --layers $LAYER --seed 1000
  done
done