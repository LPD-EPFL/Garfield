# AggregaThor

* Single server, multiuple workers setup.

* Deployment requires running `trainer.py` on multiple machines.

```
usage: trainer.py [-h] [--master MASTER] [--rank RANK] [--dataset DATASET]
                  [--batch BATCH] [--num_workers NUM_WORKERS]
                  [--fw FW] [--model MODEL]
                  [--loss LOSS] [--optimizer OPTIMIZER]
                  [--opt_args OPT_ARGS] [--num_iter NUM_ITER] [--gar GAR]
                  [--acc_freq ACC_FREQ] [--bench BENCH] [--log LOG]

AggregaThor implementation using Garfield library

optional arguments:
  -h, --help            Show this help message and exit.
  --master MASTER       Master node in the deployment. This node takes rank 0, usually the PS.
  --rank RANK           Rank of a process in the distributed setup.
  --dataset DATASET     Dataset to be used, e.g., mnist, cifar10.
  --batch BATCH         Minibatch size to be employed by each worker.
  --num_workers NUM_WORKERS
                        Number of workers in the deployment.
  --fw FW               Number of declared Byzantine workers.
  --model MODEL         Model to be trained, e.g., convnet, cifarnet, resnet.
  --loss LOSS           Loss function to optimize.
  --optimizer OPTIMIZER
                        Optimizer to use.
  --opt_args OPT_ARGS   Optimizer arguments; passed in dict format, e.g., '{"lr":"0.1"}'
  --num_iter NUM_ITER   Number of training iterations to execute.
  --gar GAR             Aggregation rule for aggregating gradients.
  --acc_freq ACC_FREQ   The frequency of computing accuracy while training.
  --bench BENCH         If True, time elapsed in each step is printed.
  --log LOG             If True, accumulated loss at each iteration is printed.

```

* We provide `run_exp.sh` to deploy and run AggregaThor on multiple machines. This script serves only as an example of how to use the provided software. The interested user can write whatever `run` script of choice. It can be used as follows:

  1. Create two files, `servers` and `workers`, and fill them with hostnames of nodes which should contribute to the experiment. The first file, `servers` should contain only one line, i.e., one host, where the second file, `workers` should contain as many lines as the number of hosts (each line should contain one hostname).

  2. Run `./run_exp.sh`. Note that the parameters to be given to `tranier.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

  3. Run `./kill.sh` to clean up.
