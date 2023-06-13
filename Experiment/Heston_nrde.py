import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import sys
import pathlib
here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..'  ))
from Solver.NRDE_Solver import PPDE_Heston_hd as PPDE
from Solver.options import Autocallable
from pynvml import *

def sample_x0(batch_size, d_s,device):
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    z = torch.randn(batch_size, d_s, device=device)
    s0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    v0 = torch.ones_like(s0) * 0.04
    x0 = torch.cat([s0,v0],1)
    return x0
    

def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


def train(T,
        n_steps,
        d,
        d_s,
        d_after,
        mu,
        vol_of_vol,
        kappa,
        theta,
        depth,
        hidden,
        ffn_hidden,
        output,
        num_layers,
        max_updates,
        batch_size, 
        lag,    
        base_dir,
        device,
        method,
        odestep,
        odesolver,
        drop,
        h,
        withx,
        d_red,
        lstm_hid,
        ncdrift
        ):    
    logfile = os.path.join(base_dir, "log.txt")
    nvmlInit()
    ts = torch.linspace(0,T,n_steps+1, device=device)
    option = Autocallable(idx_traded = 0,B=1.02,Q1=1.1,Q2=1.2,q=0.9,r=mu,ts=ts)
    ppde =  PPDE(d,d_red,lstm_hid,d_after, mu, vol_of_vol,kappa,theta, depth, hidden, ffn_hidden,output,num_layers,odestep,odesolver,withx,ncdrift,d_s)
    ppde.to(device)
    optimizer = torch.optim.Adagrad(ppde.parameters(),lr=0.1)
    
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    train_history=[]
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size,d_s, device)
        if method=="mart":
            loss, _, _ = ppde.mart_rep(ts=ts, x0=x0, option=option, lag=lag,drop=False)
        else:
            loss, _, _ = ppde.cond_exp(ts=ts, x0=x0, option=option, lag=lag,drop=False)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        # testing
        if idx%100 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000,d,device=device) 
                x0[:,1] = x0[:,1]*0.04
                if method=="mart":
                    loss, Y, payoff = ppde.mart_rep(ts=ts,x0=x0,option=option,lag=lag,drop=drop)
                else:
                    loss, Y, payoff = ppde.cond_exp(ts=ts,x0=x0,option=option,lag=lag,drop=drop)
                payoff = torch.exp(-mu*ts[-1])*payoff.mean()
                handle = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(handle)
                mem_used=info.used/(1024**3)
                train_history.append([idx+1,loss.item(),payoff.item(),Y[0,0,0].item(),mem_used])
            pbar.update(100)
            write("loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(loss.item(),payoff.item(), Y[0,0,0].item()),logfile,pbar)
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print ("Total memory:", (info.total)/(1024**3),"GB")
    print ("Used memory:", (info.used)/(1024**3),"GB")
    nvmlShutdown()
    train_history=np.array(train_history)
    np.savetxt('{}_train_history.csv'.format(base_dir),
               train_history,
               fmt=['%d', '%.5e', '%.5e', '%.5e','%.5e'],
               delimiter=",",
               header='step,loss,target_value,predicted',
               comments='')
    result = {"state":ppde.state_dict(),
            "loss":losses}
    torch.save(result, os.path.join(base_dir, "Heston_{}_test.tar".format(d)))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='../numerical_results', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--max_updates', default=200,type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--d_after', default=2, type=int)
    parser.add_argument('--ffn_hidden', default=45,type=int)
    parser.add_argument('--hidden', default=15, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--num_layers',default=3, type=int)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--odestep',default=0.025,type=float)
    parser.add_argument('--odesolver',default="midpoint",type=str)
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--vol_of_vol', default=0.05, type=float, help="vol of vol")
    parser.add_argument('--kappa', default=0.8, type=float, help="mean reverting process coef")
    parser.add_argument('--theta', default=0.3, type=float, help="mean reverting")
    parser.add_argument('--method', default="mart", type=str, help="learning method", choices=["mart","expectation"])
    parser.add_argument('--drop',default=False,type=bool)
    parser.add_argument('--h',default=0.001,type=float)
    parser.add_argument('--withx',default=False,type=bool)
    parser.add_argument('--d_red',default=False,type=bool)
    parser.add_argument('--lstm_hid',default=3,type=int)
    parser.add_argument('--ncdrift',default=False,type=bool)  
    parser.add_argument('--d_s',default=1,type=int)
    args = parser.parse_args()
    
     
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    results_path = os.path.join(args.base_dir, "Heston", args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        d_s=args.d_s,
        d_after=args.d_after,
        mu=args.mu,
        vol_of_vol=args.vol_of_vol,
        kappa=args.kappa,
        theta=args.theta,
        depth=args.depth,
        hidden=args.hidden,
        ffn_hidden=args.ffn_hidden,
        output=1,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        lag=args.lag,
        base_dir=results_path,
        device=device,
        method=args.method,
        num_layers=args.num_layers,
        odestep=args.odestep,
        odesolver=args.odesolver,
        drop=args.drop,
        h=args.h,
        withx=args.withx,
        d_red=args.d_red,
        lstm_hid=args.lstm_hid,
        ncdrift=args.ncdrift
        )
