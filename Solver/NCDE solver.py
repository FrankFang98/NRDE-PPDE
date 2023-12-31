
import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod
import numpy as np

from Solver.options import Lookback, BaseOption
from Solver.networks import *
from Solver.networks import NeuralCDE
from Solver.networks import NeuralCDE_linear
from Solver.ncde.cdeint_module import *
from Solver.ncde.interpolate import natural_cubic_spline_coeffs
from Solver.ncde.interpolate import NaturalCubicSpline


class PPDE(nn.Module):

    def __init__(self, d: int, mu: float, depth: int, hidden: int,output:int, ffn_hidden: List[int],interpolate:str):
        super().__init__()
        self.d = d
        self.mu = mu # risk free rate

        self.depth = depth
        #self.augmentations = (LeadLag(with_time=False),)
        self.sig_channels =signatory.signature_channels(channels=d, depth=depth) # x2 because we do lead-lag
        self.interpolate=interpolate
        if self.interpolate=="Cubicspline":
            self.f=NeuralCDE(input_channels=self.d+1, hidden_channels=hidden, output_channels=output, FFN=ffn_hidden)
        else:
            self.f = NeuralCDE_linear(input_channels=self.d, hidden_channels=hidden, output_channels=output,FFN=ffn_hidden ) 
        
        #self.dfdx = NeuralCDE(input_channels=self.d+1, hidden_channels=hidden, output_channels=d,FFN=ffn_hidden ) 
    @abstractmethod
    def sdeint(self, ts, x0):
        """
        Code here the SDE that the underlying assets follow
        """
        ...
    """    
    def prepare_data(self, ts: torch.Tensor, x0: torch.Tensor, lag: int):
        
        Prepare the data:
            1. Solve the sde using some sde solver on a fine time discretisation
            2. calculate path signature between consecutive timesteps of a coarser time discretisation
            3. Calculate increments of brownian motion on the coarser time discretisation
        Parameters
        ----------
        ts: torch.Tensor
            Time discrstisation. Tensor of size (n_steps + 1)
        x0: torch.Tensor
            initial value of paths. Tensor of size (batch_size, d)
        lag: int
            lag used to create the coarse time discretisation in terms of the fine time discretisation.
        
        Returns
        -------
        x: torch.Tensor
            Solution of the SDE on the fine time discretisation. Tensor of shape (batch_size, n_steps+1, d)
        path_signature: torch.Tensor
            Stream of signatures. Tensor of shape (batch_size, n_steps/lag + 1, sig_channels)
        sum_increments: torch.Tensor
            Increments of the Brownian motion on the coarse time discretisation. Tensor of shape (batch_size, n_steps/lag+1, d)
        
        x, brownian_increments = self.sdeint(ts, x0)
        device = x.device
        batch_size = x.shape[0]
        path_signature = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.sig_channels, device=device)
        sum_increments = torch.zeros(batch_size, len(range(0, len(ts), lag)), self.d, device=device)
        basepoint = torch.zeros(batch_size, 1, self.d, device=device)

        for idx, id_t in enumerate(range(0, len(ts), lag)):
            if idx == 0:
                portion_path = torch.cat([basepoint, x[:,0,:].unsqueeze(1)],1)
            else:
                portion_path = x[:,id_t-lag:id_t+1,:] 
                
            #augmented_path = apply_augmentations(portion_path, self.augmentations)
            path_signature[:,idx,:] = signatory.signature(portion_path, self.depth)
            try:
                sum_increments[:,idx,:] = torch.sum(brownian_increments[:,id_t:id_t+lag], 1)
            except:
                pass # it is the last point and we don't have anymore increments, but that's okay, because at the last step of the bsde, we compare thes olution of the bsde against the payoff of the option
        return x, path_signature, sum_increments 
    """
    
    def eval_mc(self, ts: torch.Tensor, x: torch.Tensor, lag: int, option: BaseOption,mc_samples: int):

        x = x.unsqueeze(0) if x.dim()==2 else x
        batch_size, id_t = x.shape[0], x.shape[1]
        x = torch.repeat_interleave(x, mc_samples, dim=0)
        device = x.device
        mc_paths, _ = self.sdeint(ts = ts[id_t-1:], x0 = x[:,-1,:]) 
        x = torch.cat([x, mc_paths[:,1:,:]],1)
        payoff = torch.exp(-self.mu * (ts[-1]-ts[id_t-1])) * option.payoff(x)
        payoff = payoff.reshape(batch_size, mc_samples, 1).mean(1)
        return payoff

    """    
    def fbsdeint(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int): 
        
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        option: object of class option to calculate payoff
        lag: int
            lag in fine time discretisation
        
        

        x, path_signature, brownian_increments = self.prepare_data(ts,x0,lag)
        payoff = option.payoff(x) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag]
        tx = ts[::lag].reshape(1,-1,1).repeat(batch_size,1,1)
        train_coeffs = natural_cubic_spline_coeffs(t, path_signature)
        Y = self.f(t,train_coeffs) # (batch_size, L, 1)
        Z = self.dfdx(t,train_coeffs) # (batch_size, L, dim)
        Y.permute(1,0,2)
        Z.permute(1,0,2)
        loss_fn = nn.MSELoss()
        loss = 0
        h = tx[:,1:,:] - tx[:,:-1,:] 
        discount_factor = torch.exp(-self.mu*h) # 
        target = discount_factor*Y[:,1:,:].detach()
        stoch_int = torch.sum(Z*brownian_increments,2,keepdim=True) # (batch_size, L, 1)
        pred = Y[:,:-1,:] + stoch_int[:,:-1,:] # (batch_size, L-1, 1)
        
        loss = torch.mean((pred-target)**2,0).sum()
        loss += loss_fn(Y[:,-1,:], payoff)
        
        return loss, Y, payoff
            
    """            
    def conditional_expectation(self, ts: torch.Tensor, x0: torch.Tensor, option: Lookback, lag: int,lag2:int,T:float,interpolate:str,odestep:float,odesolver:str,drop:bool): 
        """
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        option: object of class option to calculate payoff
        lag: int
            lag in fine time discretisation
        
        """
        
        x, brownian_increments = self.sdeint(ts, x0)
        payoff = option.payoff(x) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts[::lag]
        t2=ts[::lag2]
        x1=x[:,::lag2,:]
        tx=ts[::lag2].reshape(1,-1,1).repeat(batch_size,1,1)
        path= torch.cat([tx,x1],2)

        if interpolate=="Cubicspline":
            train_coeffs=natural_cubic_spline_coeffs(t2, path)
            Y=self.f(t,train_coeffs,odestep,odesolver)
        else:
            Y = self.f(ts[::lag],ts,x,lag,T) # (batch_size, L, 1)
        """
        payoff=torch.unsqueeze(payoff, 1)
        for idt in t[1:]:
            disc_pay=payoff[:,-1,:]*torch.exp(-self.mu*(idt))
            payoff=torch.cat([disc_pay.unsqueeze(1),payoff],1)
        z=torch.linalg.norm(payoff-Y,ord=float('inf'),dim=1)
        z=torch.square(z)
        loss=torch.sum(z)/batch_size
        return loss, Y, payoff
        
        
        """        
        loss_fn = nn.MSELoss()
      
        loss = 0
        for idx,idt in enumerate(ts[::lag]):
            discount_factor = torch.exp(-self.mu*(t[-1]-idt))
            target = discount_factor*payoff 
            pred = Y[:,idx,:] 
            
            loss += loss_fn(pred, target)
        
        loss=loss
        return loss, Y, payoff



class PPDE_BlackScholes(PPDE):

    def __init__(self, d: int, mu: float, sigma: float, depth: int, hidden: int, ffn_hidden: List[int],output:int,interpolate:str):
        super(PPDE_BlackScholes, self).__init__(d=d, mu=mu, depth=depth, hidden=hidden, ffn_hidden=ffn_hidden,output=output,interpolate=interpolate)
        self.sigma = sigma # change it to a torch.parameter to solve a parametric family of PPDEs
    
    
    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, N, d)
        Note
        ----
        I am assuming uncorrelated Brownian motion
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

    

class PPDE_Heston(PPDE):

    def __init__(self, d: int, mu: float, vol_of_vol: float, kappa: float, theta: float,  depth: int, hidden: int, ffn_hidden: List[int],output:int,interpolate:str):
        assert d==2, "we need d=2"
        assert 2*kappa*theta > vol_of_vol , "Feller condition is not satisfied"
        super(PPDE_Heston, self).__init__(d=d, mu=mu, depth=depth, hidden=hidden, ffn_hidden=ffn_hidden,output=output,interpolate=interpolate)
        self.vol_of_vol = vol_of_vol
        self.kappa = kappa
        self.theta = theta
    
    
    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, N, d)
        """
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts)-1, self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            s_new = x[:,-1,0] + self.mu*x[:,-1,0]*h + x[:,-1,0]*torch.sqrt(x[:,-1,1])*brownian_increments[:,idx,0]
            v_new = x[:,-1,1] + self.kappa*(self.theta-x[:,-1,1])*h + self.vol_of_vol*torch.sqrt(x[:,-1,1])*brownian_increments[:,idx,1]
            x_new = torch.stack([s_new, v_new], 1) # (batch_size, 2)
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments
