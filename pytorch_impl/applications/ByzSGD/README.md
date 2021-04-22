# ByzSGD

* Multiple servers, multiuple workers setup.

* Deployment requires running `trainer.py` on multiple machines.

```
usage: trainer.py [-h] [--master MASTER] [--rank RANK] [--dataset DATASET]
                  [--batch BATCH] [--num_ps NUM_PS]
                  [--num_workers NUM_WORKERS] [--fw FW] [--fps FPS]
                  [--model MODEL] [--loss LOSS] [--optimizer OPTIMIZER]
                  [--opt_args OPT_ARGS] [--num_iter NUM_ITER] [--gar GAR]
                  [--acc_freq ACC_FREQ] [--bench BENCH] [--log LOG]

ByzSGD implementation using Garfield library

optional arguments:
  -h, --help            Show this help message and exit.
  --master MASTER       Master node in the deployment. This node takes rank 0, usually the first PS.
  --rank RANK           Rank of a process in a distributed setup.
  --dataset DATASET     Dataset to be used, e.g., mnist, cifar10.
  --batch BATCH         Minibatch size to be employed by each worker.
  --num_ps NUM_PS       Number of parameter servers in the deployment.
  --num_workers NUM_WORKERS
                        Number of workers in the deployment.
  --fw FW               Number of declared Byzantine workers.
  --fps FPS             Number of declared Byzantine parameter servers.
  --model MODEL         Model to be trained, e.g., convnet, cifarnet, resnet.
  --loss LOSS           Loss function to optimize against.
  --optimizer OPTIMIZER
                        Optimizer to use.
  --opt_args OPT_ARGS   Optimizer arguments; passed in dict format, e.g., '{"lr":"0.1"}'
  --num_iter NUM_ITER   Number of training iterations to execute.
  --gar GAR             Aggregation rule for aggregating gradients.
  --acc_freq ACC_FREQ   The frequency of computing accuracy while training.
  --bench BENCH         If True, time elapsed in each step is printed.
  --log LOG             If True, accumulated loss at each iteration is printed.

```

* We provide `run_exp.sh` to deploy and run ByzSGD on multiple machines. This script serves only as an example of how to use the provided software. The interested user can write whatever `run` script of choice. It can be used as follows:

  1. Create two files, `servers` and `workers`, and fill them with hostnames of nodes which should contribute to the experiment. The first file, `servers` should contain (at least) one line (corresponding to the number of hosts to be used as servers), where the second file, `workers` should contain the names of the hosts that would contribute as workers in the distributed setup.

  2. Run `./run_exp.sh`. Note that the parameters to be given to `trainer.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

  3. Run `./kill.sh` to clean up.
 
## Notes on this implementation
1) The same GAR is used to aggregate both gradients and models.

2) We assume the asynchronous algorithm here.

3) Workers do not pull models from multiple parameter servers (PSes); the latter controls the learning process. In turn, the gather step among PSes is executed in each and every step.
