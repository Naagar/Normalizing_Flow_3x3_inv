# CInC Flow 

PAPER: [link](https://arxiv.org/abs/2107.01358)

Implementation of improvements for generative normalizing flows and more specifically Glow. 

We extend the 1x1 convolutions used in glow to convolutions with any kernel size and we introduce a new coupling layer.

This work is the adapted from [Emerging Convolutions for Generative Normalizing Flows](https://github.com/ehoogeboom/emerging):
```
Emiel Hoogeboom, Rianne van den Berg, and Max Welling. Emerging Convolutions for Generative Normalizing Flows. International Conference on Machine Learning, 2019.
```

## Requirements
The ```pip_installs``` script can be used to install all the required packages using pip.

## Download datasets
CIFAR10 is automatically downloaded.
Galaxy images need to be downloaded [here](https://github.com/SpaceML/merger_transfer_learning).

ImageNet 32x32 and 64x64 was downloaded from the link on the Glow github: `https://storage.googleapis.com/glow-demo/data/{dataset_name}-tfr.tar`
with `imagenet-oord` as dataset_name.

## How to start reading the code?
The quad coupling layer is defined on line 409 of the ```model.py``` file.
The convolution is defined on line 463 of the ```conv2d/conv2d.py``` file. The corresponding inversion operation can be found in ```conv2d/inverses/inverse_cython.py``` and ```conv2d\inverses\inverse_op_cython.pyx```.


## Experiments
To get infos regarding the parameter use ```python3 train.py -h```.

##### CIFAR-10 results
Emerging:
```
mpiexec -n 2 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 3 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

Glow:
```
mpiexec -n 2 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

3x3 convolution:
```
mpiexec -n 2 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 7 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

3x3 convolution and quad-coupling:
```
mpiexec -n 2 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 7 --flow_coupling 2 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --epochs 4001
```

##### ImageNet 32x32 results
This command lines assumes that the variable ```DATA_PATH``` contains the
path to the imagenet dataset.

Emerging:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 3 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

Glow:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

3x3 convolution:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 7 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

3x3 convolution and quad-coupling:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 7 --flow_coupling 2 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

##### ImageNet 64x64 results
Emerging:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 3 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

Glow:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

3x3 convolution:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 7 --flow_coupling 1 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

3x3 convolution and quad-coupling:
```
mpiexec -n 4 python3 train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 7 --flow_coupling 2 --seed 0 --learnprior --lr 0.001 --n_bits_x 8 --data_dir $DATA_PATH
```

## Sample time results
Emerging:
```
mpiexec -n 1 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 3 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --sample
```

Glow:
```
mpiexec -n 1 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --sample
```

3x3 convolution:
```
mpiexec -n 1 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 7 --flow_coupling 1 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --sample
```

3x3 convolution and quad-coupling:
```
mpiexec -n 1 python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 7 --flow_coupling 2 --seed 2 --learnprior --lr 0.001 --n_bits_x 8 --sample
