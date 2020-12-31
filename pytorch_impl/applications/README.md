# Applications

This directory includes the applications that are implemented using the Garfield library. This directory should be enriched in the future by more applications to be implemented. A high-level idea of these applications is given below. In each subdirectory, we explain how to use the corresponding application without explaining the architecture of such an application.

## Centralized

This directory implements centralized training using Garfield components. Such an example works as a perfect first-time-tutorial to get familiar with Garfield. Training is done here on one machine which owns everything: the model, the optimizer, the dataset, ..etc. 

## AggregaThor

[AggregaThor](https://infoscience.epfl.ch/record/265684) is the first published, scalable Byzantine-resilient system for machine learning applications. It was built originally on top of TensorFlow by adding robust aggregation to the shared graph employed by TensorFlow. AggregaThor relies on a `single server, multiple workers` architecture (following the parameter server architecture): it assumes one trusted, highly-available server and bunch of workers, where some of the latter could be Byzantine (behave arbitrarily). 
This directory implements a similar design with one server and multiple workers, yet using the components of the Garfield library. The robust aggregation (done on the server side) can rely on any of the aggregation rules provided by Garfield

Official Citation: 
Damaskinos G, El Mhamdi EM, Guerraoui R, Guirguis A, Rouault S. Aggregathor: Byzantine machine learning via robust gradient aggregation. In The Conference on Systems and Machine Learning (SysML), 2019.

## ByzSGD

[ByzSGD](https://dl.acm.org/doi/abs/10.1145/3382734.3405695) is the first Byzantine-resilient machine learning protocol to tolerate Byzantine servers as well as Byzantine workers in the parameter server architecture. Essentially, ByzSGD replicates the server on multiple machines and relies on communication among the server replicas to (roughly speaking) agree on the same model. In this sense, ByzSGD relies on a `multiple servers, multiple workers` architecture. In the original paper, ByzSGD presents two algorithms: synchronous and asynchronous (with regard to the network assumption). 
This directory implements the asynchronous version of ByzSGD, allowing for multiple server replicas and multiple workers and assuming no upper bounds on the communication and computation delays.

Official Citation:
El-Mhamdi EM, Guerraoui R, Guirguis A, Hoang LN, Rouault S. Genuinely distributed byzantine machine learning. In Proceedings of the 39th Symposium on Principles of Distributed Computing 2020 Jul 31 (pp. 355-364).

## Garfield_CC

This is the original implementation of the Garfield _system_. It basically implements two of the applications in this directory (AggregaThoor and ByzSGD), yet in a non-modular way (and hence, quite harder to use). Such implementation relies on the [communication collectives](https://pytorch.org/docs/stable/distributed.html), which are supported with older versions of PyTorch (RPC full support is not available in versions before `v1.6`). Such application relies on its own `datasets.py` and `models.py`. We include the implementation here for completeness, yet we advise to use it only if a new version of PyTorch is not available.
The description of this application and the design can be found in [this paper](https://arxiv.org/abs/2010.05888v1).

Official Citation:
El-Mhamdi EM, Guerraoui R, Guirguis A, Rouault S. Garfield: System Support for Byzantine Machine Learning. arXiv preprint arXiv:2010.05888. 2020 Oct 12.

## LEARN

[LEARN](https://arxiv.org/abs/2008.00742) considers a fully decentralized setup: a bunch of devices/workers collaborate to train a model without a central server. Such devices communicate in a peer-to-peer fashion in each training step. LEARN is the first protocol to solve the decentralized learning problem in the non-iid setup: the data is not assumed to be identically nor independently distributed among the workers. To do so, LEARN sometimes requires multiple communication rounds among the workers in one training iteration.
This directory implements both the *iid* and the *non-iid* versions of LEARN, showing how to implement a peer-to-peer communication fashion, with Garfield components, and how to support multiple communication rounds per training iteration. This implementation also supports network asynchrony: no upper bound assumed on the communication and the computation delays.

Official Citation:
El-Mhamdi EM, Farhadkhani S, Guerraoui R, Guirguis A, Hoang LN, Rouault S. Collaborative Learning as an Agreement Problem. arXiv preprint arXiv:2008.00742. 2020 Aug 3.

## Benchmarks

This directory does not include a separate application per se, yet it includes some micro-benchmarks for the individual aggregation rules of Garfield and for the performance of the RPC calls. The structure of this subdirectory constitutes a nice playground to benchmark any component of the system. This subdirectory can be enriched with more benchmarks in the future.
