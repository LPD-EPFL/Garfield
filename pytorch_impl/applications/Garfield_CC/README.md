# Garfield implementation using the distributed runtime

* This implementation can be used to deploy both AggregaThor-like and ByzSGD-like setups.

* It relies on the distributed runtime of PyTorch, which use the communication collectiives.

* Deployment requires running `trainer.py` on multiple machines.

```
usage: trainer.py [-h] [--master MASTER] [--rank RANK] [--dataset DATASET]
                  [--batch BATCH] [--num_ps NUM_PS]
                  [--num_workers NUM_WORKERS] [--fw FW] [--fps FPS]
                  [--model MODEL] [--loss LOSS] [--lr LR]
                  [--momentum MOMENTUM] [--wd WD] [--epochs EPOCHS]
                  [--aggregator AGGREGATOR] [--mar MAR] [--backend BACKEND]
                  [--bench BENCH] [--log LOG]

Garfield_CC: Distributed SGD playground

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
  --loss LOSS           Loss function to optimize.
  --lr LR               Initial learning rate.
  --momentum MOMENTUM   Momentum of learning.
  --wd WD               Learning weight decay.
  --epochs EPOCHS       Number of training epochs to execute.
  --aggregator AGGREGATOR
                        Aggregation rule for aggregating gradients at the parameter server side. Put 'vanilla' for the native averaging.
  --mar MAR             Aggregation rule for aggregating models at both sides: parameter servers and workers. Put 'vanilla' for the native averaging.
  --backend BACKEND     Backend for communication. This should be 'gloo' or 'nccl'.
  --bench BENCH         If True, time elapsed in each step is printed.
  --log LOG             If True, accumulated loss at each iteration is printed.
```

* We provide `run_exp.sh` to deploy and run this application on multiple machines. This script serves only as an example of how to use the provided software. The interested user can write whatever `run` script of choice. It can be used as follows:

  1. Create two files, `servers` and `workers`, and fill them with hostnames of nodes that should contribute to the experiment. The first file, `servers` should contain (at least) one line (corresponding to the number of hosts to be used as servers), where the second file, `workers` should contain the names of the hosts that would contribute as workers in the distributed setup.

  2. Run `./run_exp.sh`. Note that the parameters to be given to `tranier.py` are hard-coded in this script file. Users should feel free to change them to their favorable choices.

  3. Run `./kill.sh` to clean up.
 
## Notes on this implementation
1) We assume the asynchronous algorithm here, yet this implementation does not assume actual asynchrony, i.e., if one node does not respond (e.g., crashes), the requester node will wait indefinitely.

2) Workers are not passive in this implementation: they _pull_ models from multiple parameter servers at the beginning of each training step.
