import torch
import torch.nn as nn
from collections import namedtuple
from typing import Tuple

from Solver.ncde.cdeint_module import cdeint
from Solver.ncde.interpolate import natural_cubic_spline_coeffs
from Solver.ncde.interpolate import NaturalCubicSpline
from Solver.ncde.interpolate import linear_interpolate




class RNN(nn.Module):

    def __init__(self, rnn_in, rnn_hidden, ffn_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_in, hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True)
        layers = []
        for j in range(len(ffn_sizes)-1):
            layers.append(nn.Linear(ffn_sizes[j], ffn_sizes[j+1]))
            if j<(len(ffn_sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.ffn = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self, *args):
        """Forward method 
        
        Parameters
        ----------
        x : torch.Tensor
            Sequential input. Tensor of size (N,L,d) where N is batch size, L is lenght of the sequence, and d is dimension of the path
        Returns
        -------
        output : torch.Tensor
            Sequential output. Tensor of size (N, L, d_out) containing the output from the last layer of the RNN for each timestep
        """
        x = torch.cat(args, -1)
        output_RNN, _ = self.rnn(x)
        output = self.ffn(output_RNN)
        return output

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,FFN,activation=nn.ReLU, output_activation=nn.Tanh):
        
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        layers=[]
        for i in range(len(FFN)-1):
            if i==0:
                layers.append(torch.nn.Linear(hidden_channels,FFN[0]))
                layers.append(activation())
            
            else:
                layers.append(torch.nn.Linear(FFN[i],FFN[i+1]))
                layers.append(activation())
        layers.append(torch.nn.Linear(FFN[-1],input_channels*hidden_channels))
        layers.append(output_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        z = self.net(z)
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z
    
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels,FFN):
        super(NeuralCDE, self).__init__()
        self.hidden_channels = hidden_channels
        self.func = CDEFunc(input_channels, hidden_channels,FFN)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout_1 = torch.nn.Linear(hidden_channels, output_channels)
    
    def forward(self, times, coeffs,odestep,odesolver):
        spline = NaturalCubicSpline(times, coeffs)
        z0 = self.initial(spline.evaluate(times[0]))

    
        z_t = cdeint(dX_dt=spline.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=times,
                                   step=odestep,
                                   solver=odesolver,
                                   atol=1e-5,
                                   rtol=1e-5)
        for i in range(len(z_t.shape) - 2, 0, -1):
            z_t = z_t.transpose(0, i)
        pred_y=self.readout_1(z_t)
        
    
        return pred_y
    
class NeuralCDE_linear(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels,FFN):
        super(NeuralCDE_linear, self).__init__()
        self.hidden_channels = hidden_channels
        self.func = CDEFunc(input_channels, hidden_channels,FFN)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout_1 = torch.nn.Linear(hidden_channels, output_channels)
        
    def forward(self, ts,times,X,lag,T ):
        linear=linear_interpolate(ts,times,X,lag,T)
        z0 = self.initial(X[:,0,:])

    
        z_t = cdeint(dX_dt=linear.derivative,
                                   z0=z0,
                                   func=self.func,
                                   t=ts,
                                   atol=1e-4,
                                   rtol=1e-4)
        for i in range(len(z_t.shape) - 2, 0, -1):
            z_t = z_t.transpose(0, i)
        pred_y=self.readout_1(z_t)
        
    
        return pred_y
    