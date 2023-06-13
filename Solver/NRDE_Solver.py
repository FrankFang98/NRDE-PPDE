import os
import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod
from Solver.options import Lookback, BaseOption
import Solver.nrde as nrde
import numpy as np
from Solver.networks import RNN


class PPDE(nn.Module):

    def __init__(self, d: int,d_red:bool, lstm_hid, d_after:int, mu: float, depth: int, hidden: int,output:int, ffn_hidden: int,num_layers:int,odestep:float,odesolver:str,withx:bool,ncdrift:bool):
        super().__init__()
        self.d = d #dimension of the X
        self.d_after=d_after  # dimension after the embedding layer
        self.mu = mu #drift coefficient in the SDE
        self.withx=withx #Specify whether we use NRDE model that also include X_t dimension during each update
        self.depth = depth #depth of log-sinature
        self.d_red=d_red #Boolean value specify whether we use embedding layer to reduce the dimension or not
        self.ncdrift=ncdrift
        if self.d_red:
            self.sig_channels = signatory.logsignature_channels(in_channels=self.d_after+1, depth=depth) # x2 because we do lead-lag
            self.dim_red=RNN(rnn_in=self.d,rnn_hidden=lstm_hid,ffn_sizes=[lstm_hid]+[self.d_after])
            self.f = nrde.model.NeuralRDE(initial_dim=self.d_after+1, logsig_dim=self.sig_channels, hidden_dim=hidden, output_dim=output, num_layers=num_layers,hidden_hidden_dim=ffn_hidden,solver=odesolver,odestep=odestep) 
            self.dfdx= nrde.model.NeuralRDE(initial_dim=self.d_after+1, logsig_dim=self.sig_channels, hidden_dim=hidden, output_dim=self.d, num_layers=num_layers,hidden_hidden_dim=ffn_hidden,solver=odesolver,odestep=odestep)
        else:
            self.sig_channels = signatory.logsignature_channels(in_channels=self.d+1, depth=depth)
            self.f = nrde.model.NeuralRDE(initial_dim=self.d+1, logsig_dim=self.sig_channels, hidden_dim=hidden, output_dim=output, num_layers=num_layers,hidden_hidden_dim=ffn_hidden,solver=odesolver,odestep=odestep) 
            self.dfdx= nrde.model.NeuralRDE(initial_dim=self.d+1, logsig_dim=self.sig_channels, hidden_dim=hidden, output_dim=self.d, num_layers=num_layers,hidden_hidden_dim=ffn_hidden,solver=odesolver,odestep=odestep)
            
    @abstractmethod
    def sde_gen(self, ts, x0):
        """
        The numerical scheme we use to generate the underlying SDE of the specific model.
        """
        ...

    def prepare_data(self, ts: torch.Tensor, x0: torch.Tensor, lag: int,drop:bool):
        """
        Prepare the relavent data for NRDE network:
            1. Simulate path trajectories on the finer grid.
            2. Compute the logsignature between coarse grid for each simulated path.
            3. Calculate increments of brownian motion on the coarse intervals.
        Parameters
        ----------
        ts: The torch.Tensor represent finer time grid.
        x0: The torch.Tensor represent initial value of SDE.
        lag: lag in fine time discretisation to create coarser grid.
        
        Returns
        -------
        x:Simulated paths on finer grid.
        path_signature: Stream of logsignatures on consecutive coarse intervals.
        sum_increments: Increment of Brownian motion on consecutive coarse intervals.
        """
        if self.ncdrift:
            x, brownian_increments = self.sde_gen(ts, x=x0)
        else:
            x, brownian_increments = self.sde_gen(ts, x0)
        device = x.device
        batch_size = x.shape[0]
        if self.d_red:
            x_copy=torch.clone(x)
            x=self.dim_red(x)
        path_signature = torch.zeros(batch_size, len(range(0, len(ts), lag))-1, self.sig_channels, device=device)
        sum_increments = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.d, device=device)

        for idx, id_t in enumerate(range(0, len(ts), lag)):
            if idx == 0:
                pass
            else:
                portion_path = x[:,id_t-lag:id_t+1,:] 
                t=ts[id_t-lag:id_t+1].reshape(1,-1,1).repeat(batch_size,1,1)
                portion_path =torch.cat([t,portion_path],2)
                path_signature[:,idx-1,:] = signatory.logsignature(portion_path, self.depth)
            
            try:
                sum_increments[:,idx,:] = torch.sum(brownian_increments[:,id_t:id_t+lag], 1)
            except:
                pass 
        if self.d_red:
            return x_copy,x, path_signature, sum_increments 
        return x, path_signature, sum_increments
    
    

    
    def eval_mc(self, ts: torch.Tensor, x: torch.Tensor, lag: int, option: BaseOption, mc_samples: int):
        """
        Given a truncated trajectory up to time t<T, we use the Monte-Carlo method used to simulate the trajectory from 
        t to T and calculate the conditional expectation as the solution of PPDE at time t.
    
        """
        x = x.unsqueeze(0) if x.dim()==2 else x
        batch_size, id_t = x.shape[0], x.shape[1]
        x = torch.repeat_interleave(x, mc_samples, dim=0)
        device = x.device
        mc_paths, _ = self.sde_gen(ts = ts[id_t-1:], x0 = x[:,-1,:]) 
        x = torch.cat([x, mc_paths[:,1:,:]],1)
        payoff = torch.exp(-self.mu * (ts[-1]-ts[id_t-1])) * option.payoff(x)
        payoff = payoff.reshape(batch_size, mc_samples, 1).mean(1)
        return payoff
    
            
    
    def mart_rep(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int,drop): 
        """
        Learning the solution of PPDE using martingale representation.
        ----------
        ts: The torch.Tensor represent finer time grid.
        x0: The torch.Tensor represent initial value of SDE.
        option: The final payoff function
        lag: lag in fine time discretisation
        drop:Boolean value indicate whether we drop some of the generated data.
        
        """

        if self.d_red:
            x_copy,x, path_signature, brownian_increments = self.prepare_data(ts,x0,lag,drop)
            payoff = option.payoff(x_copy) # (batch_size, 1)
        else:
            x,path_signature,brownian_increments=self.prepare_data(ts, x0, lag, drop)
            payoff=option.payoff(x)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag]
        tx=ts[::lag].reshape(1,-1,1).repeat(batch_size,1,1)
        x1=x[:,::lag,:]
        t0=torch.zeros(batch_size,len(t),1,device=device)
        x1=torch.cat([t0,x1],2) 
        Y = self.f((x1[:,0,:],path_signature))
        Z = self.dfdx((x1[:,0,:],path_signature)) 
        

        loss_fn = nn.MSELoss()
        loss = 0
        h = tx[:,1:,:] 
        discount_factor = torch.exp(-self.mu*h) # 
        target = discount_factor*Y[:,1:,:].detach()
        stoch_int = torch.sum(torch.exp(-self.mu*tx)*Z*brownian_increments,2,keepdim=True) 
        pred = torch.exp(-self.mu*tx[:,:-1,:])*Y[:,:-1,:] + stoch_int[:,:-1,:]
        loss = torch.mean((pred-target)**2,0).sum()
        loss += loss_fn(Y[:,-1,:], payoff)
        return loss, Y, payoff
    
    

    
    def cond_exp(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int,drop:bool): 
        """
        Solving the PPDE using conditional expectation
        
        """
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

    




class PPDE_BlackScholes(PPDE):

    def __init__(self, d: int,d_red:bool,lstm_hid:int,d_after:int, mu: float, sigma: float, depth: int, hidden: int, ffn_hidden:int,output:int,num_layers:int,odestep:float,odesolver:str,withx:bool,ncdrift:bool):
        super(PPDE_BlackScholes, self).__init__(d=d,d_red=d_red,lstm_hid=lstm_hid,d_after=d_after, mu=mu, depth=depth, hidden=hidden, ffn_hidden=ffn_hidden,output=output,num_layers=num_layers,odestep=odestep,odesolver=odesolver,withx=withx,ncdrift=ncdrift)
        self.sigma = sigma # change it to a torch.parameter to solve a parametric family of PPDEs
    
    
    def sde_gen(self, ts, x0):
        """
        We use Euler scheme here to generate sample paths, and we assume the Brownian motions are uncorrelated.
        """
    
        x = x0.unsqueeze(1) 
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts)-1, self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            x_new = x[:,-1,:] + self.mu*x[:,-1,:]*h + self.sigma*x[:,-1,:]*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments

    
class PPDE_Heston_hd(PPDE):

    def __init__(self, d: int,d_red:bool,lstm_hid:int,d_after:int,  mu: float, vol_of_vol: float, kappa: float, theta: float,  depth: int,  hidden: int, ffn_hidden:int,output:int,num_layers:int,odestep:float,odesolver:str,withx:bool,ncdrift:bool,d_s:int):
    
        assert 2*kappa*theta > vol_of_vol , "Feller condition is not satisfied"
        super(PPDE_Heston_hd, self).__init__(d=d,d_red=d_red,lstm_hid=lstm_hid,d_after=d_after, mu=mu, depth=depth, hidden=hidden, ffn_hidden=ffn_hidden,output=output,num_layers=num_layers,odestep=odestep,odesolver=odesolver,withx=withx,ncdrift=ncdrift)
        self.vol_of_vol = vol_of_vol
        self.kappa = kappa
        self.theta = theta
        self.d_s=d_s
    def sde_gen(self, ts, x0):
        """
        Euler scheme to generate underlying trajectories of SDE.
        """
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts)-1, self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            s_new = x[:,-1,:self.d_s] + self.mu*x[:,-1,:self.d_s]*h + x[:,-1,:self.d_s]*torch.sqrt(x[:,-1,self.d_s:])*brownian_increments[:,idx,:self.d_s]
            v_new = x[:,-1,self.d_s:] + self.kappa*(self.theta-x[:,-1,self.d_s:])*h + self.vol_of_vol*torch.sqrt(x[:,-1,self.d_s:])*brownian_increments[:,idx,self.d_s:]
            x_new = torch.cat([s_new, v_new], 1) # (batch_size, 2)
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments

class PPDE_heat(PPDE):
    def __init__(self,  d: int,d_red:bool,lstm_hid:int,d_after:int, mu: float, sigma: float, depth: int, hidden: int, ffn_hidden:int,output:int,num_layers:int,odestep:float,odesolver:str,withx:bool,ncdrift:bool):
        super(PPDE_heat,self).__init__(d=d,d_red=d_red,lstm_hid=lstm_hid,d_after=d_after,mu=mu, depth=depth, hidden=hidden, output=output, ffn_hidden=ffn_hidden, num_layers=num_layers, odestep=odestep, odesolver=odesolver, withx=withx, ncdrift=ncdrift)
        self.sigma=sigma
    def sde_gen(self, ts,x0):
        """
        Here the underlying SDE is just Brownian motion, so everything is straigh forward.
        """

        
        x = x0.unsqueeze(1) 
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts)-1, self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            x_new=x[:,-1,:]+self.sigma*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments