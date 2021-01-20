# Garfield-TF
The implementation of the Garfield library on TensorFlow. [Garfield](https://arxiv.org/abs/2010.05888) gives support for building Byzantine-resilient machine learning applications.

Code authors: Arsany Guirguis, Anton Ragot, and Jérémy Plassmann.

## Requirements
Garfield was tested with the following versions
* grpcio (1.33.2)
* numpy (1.19.1)
* python (3.8)
* tensorflow (2.3.1)
* tensorflow-datasets (4.0.1)

## Structure

* `libs/`

   The main components of the library, enabling easy sharing amongst the applications.

* `applications/`

   Examples on applications that can be built by Garfield. This directory can be enriched by other applications in the future.

   Each subdirectory corresponds to one application (*SSMW, MMSW, LEARN*).

   Each application subdirectory should contain:
   
   1. _Symlinks_ to the `libs/` and `rsrcs/` directories.

   2. `run.sh`: an example script to automatically run the corresponding application on multiple nodes.

   3. `config_generator.py`: python script that creates the configuration files with the entry of the user.

   4. `trainer.py`: the implementation of the corresponding application using the Garfield library.

   5. `config/`: directory containing the configuration files of the cluster.

* `rsrcs/`

   Other reusable resources.



## Usage

In order to run a process, you need to have a `TF_CONFIG` file for _each_ process in the cluster.

### Setting TF_CONFIG file

This file inform a process of the different nodes in the cluster, as well as its task.

```
{
  "cluster": {
      "worker": ["host0:port0", "host1:port1", "host2:port2"],
      "ps": ["host3:port3", "host4:port4"]
  },

  "task": {
    "type": "ps",
    "index": 0,
    "strategy_gradient": "Average",  (Aggregation to use on models)
    "strategy_model": "Median",      (Aggregation to use on gradients)
    "attack": "None"                 (Define if the process is bizantine or not)
  }
}
```

##### Cluster declaration:

This part needs to be the same for all config file in the cluster.

```
"cluster": {
    "worker": ["host0:port0", "host1:port1", "host2:port2"],
    "ps": ["host3:port3", "host4:port4"]
}
```

##### Task declaration:

This part must be unique for each process. You need to define if your process is a `ps` or a `worker`. The index informs the position of the process's ip in the IP:PORT list (starting at 0).
```
"task": {
    "type": "ps",
    "index": 0,
    "strategy_gradient": "Average", 
    "strategy_model": "Median", 
    "attack": "None"    
}
```

##### Aggregation and attacks:

Aggregations and attacks can only be chosen from a specific list. Note that every name is case sensitive.

***Aggregation*** : Average, Median, Krum, Brute, Aksel, Condense, Bulyan.

***Attacks*** : Random, Reverse, PartialDrop, LittleIsEnough, FallEmpires.

### Starting a process

Run `trainer.py` to start a process.

```
usage: trainer.py [-h] --config CONFIG [--log] [--max_iter MAX_ITER]
                 [--dataset DATASET]
                 [--model MODEL]
                 [--batch_size BATCH_SIZE]
                 [--nbbyzwrks NBBYZWRKS]
                 [--native]
                 
arguments:
  -h, --help                show this help message and exit
  --config CONFIG           config file location.
  --log                     display intermediary steps
  --max_iter MAX_ITER       maximum number of epoch
  --dataset DATASET         choose the dataset to use
  --model MODEL             choose the model to use
  --batch_size BATCH_SIZE   set the batch size
  --nbbyzwrks NBBYZWRKS     set the number of byzantine workers (necessary for Krum aggregation)
  --native                  choose to use the native aggregators

```

## Useuful practical general notes
1. The repo should be cloned on all nodes contributing to the experiment.

2. The bash scripts (`run.sh`) require **password-less** ssh access among machines contributing to the distributed setup.

3. `run.sh` first runs the configuration file generator and then start the processes on the different machines. The distributed setup is assumed to share the file system.

4. Our experiments so far are all done on [Grid5000](https://www.grid5000.fr). The GPU-based experiments are held on the *Lille* site.

## Requirements

All requirements are listed in the [requirement file](https://github.com/LPD-EPFL/Garfield_TF/blob/main/requirements.txt)
