# Benchmarks

* This directory includes scripts to benchmark (1) individual gradient aggregation rules (GARs) and (2) RPC calls.

* Such directory can be enriched with more benchmarks in the future.

## GARs benchmark

* The user can run `gar_bench.py` to benchmark the existing GARs provided in the Garfield library. Example: run `python gar_bench.py`.

* Currently, we support benchmarking with different values for _n_, the number of input gradients, _f_, the number of declared Byzantine inputs, and _d_, the input dimension. The benchmarking parameters currently are hard-coded in the python script. The user should feel free to dig in and choose favorable parameters.

* Note that: unlike the RPC benchmark, the GARs benchmark does not require running on multiple machines; running the python script on one machine suffices.

## RPC benchmark

* This benchmark requires running `rpc_bench.py` on multiple machines.

```
usage: rpc_bench.py [-h] [--master MASTER] [--rank RANK] [--d D] [--n N]
                    [--num_iter NUM_ITER]

Benchmarking RPC calls in the Garfield framework

optional arguments:
  -h, --help           Show this help message and exit.
  --master MASTER      Master node in the deployment. This node takes rank 0, usually the first node.
  --rank RANK          Rank of a process in a distributed setup.
  --d D                Simulated model dimension.
  --n N                Number of nodes in the deployment.
  --num_iter NUM_ITER  Number of RPC calls to do (for statistical variance).
```

* We provide `run_rpc_bench.sh n d` to deploy and run this benchmark on multiple machines: _n_ denotes the number of nodes in the deployment, and _d_ denotes the simulated model dimension. This script serves only as an example of how to use the provided software. The interested user can write whatever `run` script of choice. It can be used as follows:

  1. Create one file, `nodes`, and fill it with hostnames of nodes that should contribute to the experiment: one line for each hostname.

  2. Run `run_rpc_bench.sh 3 10000`: this will run the RPC benchmark, assuming 3 nodes in the network (communicating in a peer-to-peer fashion) and a model of size 1000 parameters.

  3. Run `./kill.sh` to clean up.
