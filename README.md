# Neural RDE Solver for Path-Dependent Partial Differential Equations [[Stochastic Analysis and Applications 2025](https://doi.org/10.1007/978-3-032-03914-9_9)]
## Overview 
This repository contains the implementation for a neural rough differential equation (NRDE) based solver for path-dependent partial differential equations (PPDEs). The paper investigates PPDEs of the form
```math
\begin{aligned}
& {\left[\partial_t u+b D_x u+\frac{1}{2} tr\left[D_{x x} u \sigma^T \sigma\right]-r u\right](t, \omega)+f(t, \omega)=0} \\
& u(T, \omega)=g(\omega), \quad t \in[0, T], \quad \omega \in C\left([0, T] ; \mathbb{R}^d\right),
\end{aligned}
```
where $u:[0, T] \times C\left([0, T], \mathbb{R}^d\right) \rightarrow \mathbb{R}$ is the path-dependent solution of the PPDE, and $b,\sigma,r,f$ are functions with corresponding dimensions. By the functional Feynman-Kac formula, the PPDE solution can be represented as a conditional expectation. This motivates neural-network approximation methods, as studied in previous work such as [[Deep-PPDE](https://github.com/msabvid/Deep-PPDE)]. In this paper, we propose a solver based on a neural rough differential equation network [[NRDE](https://github.com/jambo6/neuralRDEs/tree/master)]. NRDEs are well suited to long time series because they use log-signatures as path representations. The proposed model uses the memory efficiency of the NRDE architecture, and an additional embedding layer improves scalability as dimensionality increases. ![EL-NRDE solver](https://github.com/FrankFang98/NRDE-PPDE/blob/main/EL-NRDE%20solver.png)



## Libraries and dependencies
We recommend checking the following repositories related to Neural-ODE (NODE) type models and previous work on solving PPDEs using neural networks:
- torchdiffeq [[torchdiffeq](https://github.com/rtqichen/torchdiffeq)]: differentiable ODE solvers with full GPU support and adjoint backpropagation using $\mathcal{O}(1)$ memory.
- Neural CDE [[torchcde](https://github.com/patrick-kidger/torchcde)]: neural controlled differential equation networks that generalise NODEs and incorporate sequential input. 
- Neural RDE [[NRDE](https://github.com/jambo6/neuralRDEs/tree/master)]: the neural rough differential equation network used in our paper.
- Deep PPDE [[Deep-PPDE](https://github.com/msabvid/Deep-PPDE)]: an LSTM network with signature input for approximating PPDE solutions, used as our baseline.

### Set up the environment
Install the required packages by running:
- `pip install -r requirement.txt`
- `pip install signatory==1.2.6.1.7.1 --no-cache-dir --force-reinstall`
`signatory` should be installed after the corresponding PyTorch version has been installed; otherwise, installation errors may occur. 

## Structure of the code
### Solver
The code for the NRDE solver for PPDEs is in the [NRDE_Solver.py](https://github.com/FrankFang98/NRDE-PPDE/blob/main/Solver/NRDE_Solver.py) file under the [Solver](https://github.com/FrankFang98/NRDE-PPDE/tree/main/Solver) folder. This folder also includes the general neural RDE package used to build the solver.
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
The [Experiment](https://github.com/FrankFang98/NRDE-PPDE/tree/main/Experiment) folder contains the three experiments used in the paper. For example, to run the heat-equation experiment with dimension $d=4$ and no embedding layer, 
- `python Heat_nrde.py --d 4 --d_red False`
After training, model parameters are stored in the [numerical_results](https://github.com/FrankFang98/NRDE-PPDE/tree/main/numerical_results) folder.
```Python
result = {"state":ppde.state_dict(),
            "loss":losses}
torch.save(result, os.path.join(base_dir, "model_{}.tar".format(d)))
```
### Numerical Results
This folder contains the trained models for each experiment. The `.tar` files contain model parameters that can be loaded using `torch.load` and `model.load_state_dict` once the model hyperparameters are specified. The notebook [Report](https://github.com/FrankFang98/NRDE-PPDE/blob/main/Report.ipynb) illustrates how to load the trained models and reproduce plots and results from the paper.

## Citation
```
@inproceedings{fang2024neural,
  title={A Neural RDE-Based Model for Solving Path-Dependent Parabolic PDEs},
  author={Fang, Bowen and Ni, Hao and Wu, Yue},
  booktitle={Conference on Modern Topics in Stochastic Analysis and Applications (in honour of Terry Lyons’ 70th birthday)},
  pages={231--271},
  year={2024},
  organization={Springer}
}
```
