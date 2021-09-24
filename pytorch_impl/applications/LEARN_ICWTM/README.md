# LEARN_ICWTM

* Fully decentralized setup, yet with using coordinate-wise trimmed-mean as a GAR.

* Deployment requires running `trainer.py` on multiple machines.

```
usage: trainer.py [-h] [--master MASTER] [--rank RANK] [--dataset DATASET]
                  [--batch BATCH] [--num_nodes NUM_NODES] [--f F]
                  [--model MODEL] [--loss LOSS] [--optimizer OPTIMIZER]
                  [--opt_args OPT_ARGS] [--num_iter NUM_ITER] [--gar GAR]
                  [--acc_freq ACC_FREQ] [--non_iid NON_IID] [--bench BENCH]
                  [--log LOG]

LEARN implementation using Garfield library

optional arguments:
  -h, --help            Show this help message and exit.
  --master MASTER       Master node in the deployment. This node takes rank 0, usually the first node.
  --rank RANK           Rank of a process in the distributed setup.
  --dataset DATASET     Dataset to be used, e.g., mnist, cifar10.
  --batch BATCH         Minibatch size to be employed by each node.
  --num_nodes NUM_NODES
                        Number of nodes in the deployment.
  --f F                 Number of declared Byzantine workers.
  --model MODEL         Model to be trained, e.g., convnet, cifarnet, resnet.
  --loss LOSS           Loss function to optimize.
  --optimizer OPTIMIZER
                        Optimizer to use.
  --opt_args OPT_ARGS   Optimizer arguments; passed in dict format, e.g., '{"lr":"0.1"}'
  --num_iter NUM_ITER   Number of training iterations to execute.
  --gar GAR             Aggregation rule for aggregating gradients.
  --acc_freq ACC_FREQ   The frequency of computing accuracy while training.
  --non_iid NON_IID     If True, the algorithm with non-iid data will be executed; otherwise, iid data will be assumed. Note: this flag does not itslef change the distriibution of data over workers.
  --bench BENCH         If True, time elapsed in each step is printed.
  --log LOG             If True, accumulated loss at each iteration is printed.

```

* We provide `run_exp.sh` to deploy and run LEARN on multiple machines. This script serves only as an example on how to use the provided software. The interested user can write whatever `run` script of choice. It can be used as follows:

  1. Create one file, `nodes`, and fill it with hostnames of nodes which should contribute to the experiment: one line for each hostname.

  2. Run `./run_exp.sh`. Note that the parameters to be given to `tranier.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

  3. Run `./kill.sh` to cleanup.
