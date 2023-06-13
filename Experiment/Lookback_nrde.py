import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import matplotlib.pyplot as plt
import pathlib
import sys
here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..'  ))




from Solver.options import Lookback
from pynvml import *


def sample_x0(batch_size, dim, device):
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    z = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    return x0
    

def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


def train(T,
        n_steps,
        d,
        d_after,
        mu,
        sigma,
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
    if ncdrift:
        from Solver.NRDE_Solver import PPDE_BlackScholes_ncdrift as PPDE
    else:
        from Solver.NRDE_Solver import PPDE_BlackScholes as PPDE
    nvmlInit()
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0,T,n_steps+1, device=device)
    lookback = Lookback()
    ppde = PPDE(d,d_red,lstm_hid,d_after, mu, sigma, depth, hidden, ffn_hidden,output,num_layers,odestep,odesolver,withx,ncdrift)
    ppde.to(device)
    optimizer = torch.optim.Adagrad(ppde.parameters(),lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    train_history=[]
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device)
        if method=="mart":
            loss, _, _ = ppde.mart_rep(ts=ts, x0=x0, option=lookback, lag=lag,drop=drop)
        else:
            loss, _, _ = ppde.cond_exp(ts=ts, x0=x0, option=lookback, lag=lag,drop=drop)
        loss.backward()
        optimizer.step() 
        scheduler.step()
        losses.append(loss.detach().cpu().item())
        # testing
        if (idx+1) % 100 == 0 or idx==0:
            with torch.no_grad():
                x0 = torch.ones(3000,d,device=device) # we do monte carlo
                if method=="mart":
                    loss, Y, payoff = ppde.mart_rep(ts=ts,x0=x0,option=lookback,lag=lag,drop=drop)
                else:
                    loss, Y, payoff = ppde.cond_exp(ts=ts,x0=x0,option=lookback,lag=lag,drop=drop)
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
    
    
    result = {"state":ppde.state_dict(),
            "loss":losses}
    torch.save(result, os.path.join(base_dir, "model_{}_test.tar".format(d)))

    






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='../numerical_results', type=str)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--max_updates', default=200,type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d', default=4, type=int)
    parser.add_argument('--d_after', default=2, type=int)
    parser.add_argument('--ffn_hidden', default=30,type=int)
    parser.add_argument('--hidden', default=15, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--num_layers',default=6, type=int)
    parser.add_argument('--T', default=1.0, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--odestep',default=0.025,type=float)
    parser.add_argument('--odesolver',default="midpoint",type=str)
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=0.3, type=float, help="risk free rate")
    parser.add_argument('--method', default='mart', type=str, help="learning method", choices=["mart","expectation"])
    parser.add_argument('--drop',default=False, type=bool)
    parser.add_argument('--h',default=0.001,type=float)
    parser.add_argument('--withx',default=False,type=bool)
    parser.add_argument('--d_red',default=False,type=bool)
    parser.add_argument('--lstm_hid',default=15,type=int)
    parser.add_argument('--ncdrift',default=False,type=bool)
    args = parser.parse_args()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    results_path = os.path.join(args.base_dir, "BS", args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        d_after=args.d_after,
        mu=args.mu,
        sigma=args.sigma,
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
