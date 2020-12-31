# Centralized

* An implementation for centralized training (on one machine only) using Garfield library.

* One can run `trainer.py`, which can be used as follows.

```
usage: trainer.py [-h] [--dataset DATASET] [--batch BATCH] [--model MODEL]
                  [--loss LOSS] [--optimizer OPTIMIZER] [--opt_args OPT_ARGS]
                  [--num_iter NUM_ITER] [--acc_freq ACC_FREQ] [--bench BENCH]
                  [--log LOG]

Centralized training using Garfield library

optional arguments:
  -h, --help            Show this help message and exit.
  --dataset DATASET     Dataset to be used, e.g., mnist, cifar10.
  --batch BATCH         Minibatch size to be employed by each worker.
  --model MODEL         Model to be trained, e.g., convnet, cifarnet, resnet.
  --loss LOSS           Loss function to optimize.
  --optimizer OPTIMIZER
                        Optimizer to use.
  --opt_args OPT_ARGS   Optimizer arguments; passed in dict format, e.g., '{"lr":"0.1"}'
  --num_iter NUM_ITER   Number of training iterations to execute.
  --acc_freq ACC_FREQ   The frequency of computing accuracy while training.
  --bench BENCH         If True, time elapsed in each step is printed.
  --log LOG             If True, accumulated loss at each iteration is printed.

```

* We provide `run.sh` as an example of how to use the provided software. The interested user can write whatever `run` script of choice. Note that the parameters to be given to `tranier.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

* Example: Run `./run.sh localhost`.
