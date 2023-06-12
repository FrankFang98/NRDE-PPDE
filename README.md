# A neural rough differential equation netowrk based solver for path-dependent partial differential equation.   [[arXiv](https://arxiv.org/abs/2306.01123)]
## Overview 
In this paper we investigate the solution of path-dependent partial differential equation (PPDE) of the form
```math
\begin{aligned}
& {\left[\partial_t u+b D_x u+\frac{1}{2} tr\left[D_{x x} u \sigma^T \sigma\right]-r u\right](t, \omega)+f(t, \omega)=0} \\
& u(T, \omega)=g(\omega), \quad t \in[0, T], \quad \omega \in C\left([0, T] ; \mathbb{R}^d\right),
\end{aligned}
```
where $u:[0, T] \times C\left([0, T], \mathbb{R}^d\right) \rightarrow \mathbb{R}$ is the path-dependent solution of above PPDE, and $b,\sigma,r,f$ are functions with corresponding dimensions. By functional Feynman-Kac formula, we can write the solution of PPDE as the condtional expectation, and this open the possibility of using neural network to approximate the solution, which has been studied in previous literatures like [[Deep-PPDE](https://github.com/msabvid/Deep-PPDE)]. In this paper, we propose a solver based on neural rough differential equation network [[NRDE](https://github.com/jambo6/neuralRDEs/tree/master)], and the reason of choosing this specific network is its capability to tackle long time series efficiently by using log-signature as the path representation. We showcase that the proposed model enjoys the memory efficiency of the NRDE model and by introducing an extra embedding layer, the model is scalable with increasing dimensionality. ![EL-NRDE sovler](https://github.com/FrankFang98/PPDE-CDE/blob/main/EL-NRDE%20solver.png)



## Libraries and dependencies
We strongly recommend checking the following repositories relating to Neural-ODE (NODE) type models and previous works on solving PPDE using neural network:
- torchdiffeq [[torchdiffeq](https://github.com/rtqichen/torchdiffeq)]: A differentiable ODE solvers with full GPU support and adjoint backpropagation using $\mathcal{O}(1)$ memory.
- Neural CDE [[torchcde](https://github.com/patrick-kidger/torchcde)]: Neural controlled differential equation network that generalise NODE and incorporate sequential input. 
- Neural RDE [[NRDE](https://github.com/jambo6/neuralRDEs/tree/master)]: Neural rough differential equation network we use in our paper.
- Deep PPDE [[Deep-PPDE](https://github.com/msabvid/Deep-PPDE)]: Using LSTM network with signature input to approximate solution of PPDE.

### Set up environment
Install the required packages by running:
- `pip install -r requirement.txt`
- `pip install signatory==1.2.6.1.7.1 --no-cache-dir --force-reinstall`
The signatory have to be installed after the installation of corresponding `Pytorch` version, otherwise it may cause error message. 


