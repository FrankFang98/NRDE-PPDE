# A neural rough differential equation netowrk based solver for path-dependent partial differential equation.   [[arXiv](https://arxiv.org/abs/2306.01123)]
## Overview 
In this paper we investigate the solution of path-dependent partial differential equation (PPDE) of the form
```math
\begin{aligned}
& {\left[\partial_t u+b D_x u+\frac{1}{2} tr\left[D_{x x} u \sigma^T \sigma\right]-r u\right](t, \omega)+f(t, \omega)=0} \\
& u(T, \omega)=g(\omega), \quad t \in[0, T], \quad \omega \in C\left([0, T] ; \mathbb{R}^d\right),
\end{aligned}
```
where $u:[0, T] \times C\left([0, T], \mathbb{R}^d\right) \rightarrow \mathbb{R}$ is the path-dependent solution of above PPDE, and $b,\sigma,r,f$ are functions with corresponding dimensions. By functional Feynman-Kac formula, we can write the solution of PPDE as the condtional expectation, and this open the possibility of using neural network to approximate the solution, which has been studied in previous literatures like [[Deep-PPDE](https://github.com/msabvid/Deep-PPDE)]. In this paper, we propose a solver based on neural rough differential equation network [[NRDE](https://github.com/jambo6/neuralRDEs/tree/master)], and the reason of choosing this specific network is its capability to tackle long time series efficiently by using log-signature as the path representation. We showcase that the proposed model enjoys the memory efficiency of the NRDE model and by introducing an extra embedding layer, the model is scalable with increasing dimensionality. ![EL-NRDE sovler](https://github.com/FrankFang98/NRDE-PPDE/blob/main/EL-NRDE%20solver.png)



## Libraries and dependencies
We strongly recommend checking the following repositories relating to Neural-ODE (NODE) type models and previous works on solving PPDE using neural network:
- torchdiffeq [[torchdiffeq](https://github.com/rtqichen/torchdiffeq)]: A differentiable ODE solvers with full GPU support and adjoint backpropagation using $\mathcal{O}(1)$ memory.
- Neural CDE [[torchcde](https://github.com/patrick-kidger/torchcde)]: Neural controlled differential equation network that generalise NODE and incorporate sequential input. 
- Neural RDE [[NRDE](https://github.com/jambo6/neuralRDEs/tree/master)]: Neural rough differential equation network we use in our paper.
- Deep PPDE [[Deep-PPDE](https://github.com/msabvid/Deep-PPDE)]: Using LSTM network with signature input to approximate solution of PPDE.

### Set up the environment
Install the required packages by running:
- `pip install -r requirement.txt`
- `pip install signatory==1.2.6.1.7.1 --no-cache-dir --force-reinstall`
The signatory have to be installed after the installation of corresponding `Pytorch` version, otherwise it may cause error message. 

## Structure of the code
### Solver
The code for our NRDE solver for PPDE is in the [NRDE_Solver.py](https://github.com/FrankFang98/NRDE-PPDE/blob/main/Solver/NRDE_Solver.py) file under the [Solver](https://github.com/FrankFang98/NRDE-PPDE/tree/main/Solver) folder. In the folder we also include the package for the general neural RDE network, which we will use to build our solver.
#### Call the NRDE network
```Python
self.sig_channels = signatory.logsignature_channels(in_channels=self.d+1, depth=depth)
self.f = nrde.model.NeuralRDE(initial_dim=self.d+1, logsig_dim=self.sig_channels, hidden_dim=hidden, output_dim=output, num_layers=num_layers,hidden_hidden_dim=ffn_hidden,solver=odesolver,odestep=odestep
self.dfdx= nrde.model.NeuralRDE(initial_dim=self.d+1, logsig_dim=self.sig_channels, hidden_dim=hidden, output_dim=self.d, num_layers=num_layers,hidden_hidden_dim=ffn_hidden,solver=odesolver,odestep=odestep)
```
#### Define the objective function.
```Python
def cond_exp(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int,drop:bool): 
        if self.d_red:
            x_copy,x, path_signature, brownian_increments = self.prepare_data(ts,x0,lag,drop)
            payoff = option.payoff(x_copy) 
        else:
            x,path_signature,brownian_increments=self.prepare_data(ts, x0, lag, drop)
            payoff=option.payoff(x)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag]
        x1=x[:,::lag,:]
        t0=torch.zeros(batch_size,len(t),1,device=device)
        x1=torch.cat([t0,x1],2)
        if not self.withx:
            Y = self.f((x1[:,0,:],path_signature))
        else:
            Y = self.f((x1[:,0,:],path_signature,x[:,::lag,:]))
        loss_fn = nn.MSELoss()
        loss = 0
        for idx,idt in enumerate(ts[::lag]):
            if self.ncdrift:
                discount_factor = torch.exp(-self.mu*0.5*(t[-1]**2-idt**2))
            else:
                discount_factor = torch.exp(-self.mu*(t[-1]-idt))
            target = discount_factor*payoff 
            pred = Y[:,idx,:] 
            loss += loss_fn(pred, target)
        return loss, Y,payoff
```
### Experiment
The [Experiment](https://github.com/FrankFang98/NRDE-PPDE/tree/main/Experiment) folder contains the three experiments we conduct. For example, to run the heat equation's experiment with dimension $d=4$ and no embedding layer, 
- `python Heat_nrde.py --d 4 --d_red False`
After the training, the model parameters will be stored in the [numerical_results](https://github.com/FrankFang98/NRDE-PPDE/tree/main/numerical_results) folder.
```Python
result = {"state":ppde.state_dict(),
            "loss":losses}
torch.save(result, os.path.join(base_dir, "model_{}.tar".format(d)))
```
### Numerical result
This folder contains all the models we get for each experiment. The `.tar` file contains all the model parameters and could be loaded using `torch.load` and `model.load_state_dict` functions once we specify the hyperparameters of the model. We include a jupyter notebook [Report](https://github.com/FrankFang98/NRDE-PPDE/blob/main/Report.ipynb) in the main directory to illustrate this procedure and generate the plot and result from the paper.

## Citation
```
@misc{fang2023neural,
      title={A Neural RDE-based model for solving path-dependent PDEs}, 
      author={Bowen Fang and Hao Ni and Yue Wu},
      year={2023},
      eprint={2306.01123},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
