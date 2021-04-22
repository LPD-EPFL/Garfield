# Garfield-PT

The implementation of the Garfield library on PyTorch. [Garfield](https://arxiv.org/abs/2010.05888) gives support for 
building Byzantine-resilient machine learning applications.


## Requirements
Garfield was tested with the following versions
* torch (1.6) [tested with both cuda and non-cuda versions]
* torchvision (0.7.0)
* Python (3.6.10)
* Numpy (1.19.1)
* Scipy (1.5.2)

## Installation
The following steps should be applied for **ALL** machines that are going to contribute to the running of Garfield applications.

1. Follow [PyTorch installation guide](https://pytorch.org/) (depending on your environment and preferences). Here are couple of examples:
 * If you want to use conda, download the installer from [here](https://www.anaconda.com/products/individual) and then run `sh Anaconda_$VERSION_NO.sh; conda install python=3.6.10`. Then, install Pytorch by running: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch` (assuming using the same tested environment). Note that, for this, you need to add conda path to `PATH` environment variable as follows `export PATH=$HOME/anaconda3/bin:$PATH` then run `source activate base`. **For the distributed setup, you will need to add this latter export line in `.bashrc` file in `$HOME` directory.**
 * If you want to use pip, run `pip install torch torchvision`.

2. Install the other required packages by running `pip install numpy==1.19.1 scipy==1.5.2` (this command works also for `conda` users).

## Structure

* `libs/`

   The main components of the library, enabling easy sharing amongst the applications.

   Each subdirectory contains one main module of Garfield.

* `applications/`

   Examples on applications that can be built by Garfield. This directory can be enriched by other applications in the 
future.
   
   Each subdirectory corresponds to one application.

   Each application subdirectory should contain:

   1. _Symlinks_ to the used modules from the `libs` directory.

   2. `run.sh`: an example script to automatically run the corresponding application on multiple nodes.

   3. `kill.sh`: a script to end the deployment and do the cleaning.

   4. `trainer.py`: the implementation of the corresponding application using the Garfield library.
   
   5. `nodes` (or equivalent): the list of the machines which should contribute to training for the corresponding application.

   6. [Optional] `README.md`: a guide on how to use the application and a few notes about it.

* `rsrcs/`

   Other reusable resources.

   The playground inside is useful for trying out new stuff with the library. Once a complete application is 
implemented, it must be moved to the `applications` directory.

## Useful practical general notes
1. The repo should be cloned on all nodes contributing to the experiment. Also, all nodes should have the same `nodes` file (NFS would be a good choice for that purpose), which can be found in the running application directory.

2. The bash scripts (`run.sh` and `kill.sh`) require **password-less** ssh access among machines contributing to the distributed setup.

3. Our experiments so far are all done on [Grid5000](https://www.grid5000.fr). The GPU-based experiments are held on the *Lille* site.

4. Using inception requires 299*299 input -- see `libs/garfieldpp/datasets.py`

5. Using inception requires passing output.logits to loss() -- see `libs/garfieldpp/worker.py`

6. Use scipy v1.3.3 for faster loading of inception network


Corresponding author: Arsany Guirguis <arsany.guirguis@epfl.ch>
