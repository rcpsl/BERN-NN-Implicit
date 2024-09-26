import torch.nn as nn
from intervals.symbolic_interval import SymbolicInterval
import torch
from copy import deepcopy
class ReLU(nn.Module):
    def __init__(self, torch_layer, input_shape = None):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None #dummy for JIT
        self.post_symbolic = None  #dummy for JIT
        self.active = torch.tensor([])
        self.inactive = torch.tensor([])
        self.unstable = torch.tensor([])
        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, x: SymbolicInterval, layer_mask = None) ->SymbolicInterval:
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        post_interval = SymbolicInterval(x.input_interval, x.l.clone(), x.u.clone())

        inactive_relus_idx = torch.nonzero(x.conc_ub <= 0,as_tuple=True)
        post_interval.l[inactive_relus_idx] = 0
        post_interval.u[inactive_relus_idx] = 0
        self.inactive = inactive_relus_idx

        active_relus_idx = torch.nonzero(x.conc_lb > 0,as_tuple=True)
        post_interval.l[active_relus_idx] = x.l[active_relus_idx]
        post_interval.u[active_relus_idx] = x.u[active_relus_idx]
        self.active = active_relus_idx

        unstable_relus_idx = torch.nonzero((x.conc_lb < 0) * (x.conc_ub > 0),as_tuple = True)
        self.unstable = unstable_relus_idx
        if(len(unstable_relus_idx) != 0):

            unstable_pre_conc_lb = x.conc_lb[unstable_relus_idx]
            unstable_pre_conc_ub = x.conc_ub[unstable_relus_idx]
            # unstable_pre_max_lb = x.max_lb[unstable_relus_idx]

            #The ReLU is inactive for most of the input space
            mostly_inactive = torch.nonzero((torch.abs(unstable_pre_conc_lb) > torch.abs(unstable_pre_conc_ub)) + (x.max_lb[unstable_relus_idx] <=0), as_tuple= True)
            mostly_inactive = (unstable_relus_idx[0][mostly_inactive],unstable_relus_idx[1][mostly_inactive])
            # mostly_inactive = unstable_relus_idx[mostly_inactive]
            post_interval.l[mostly_inactive] = 0

            mostly_active = torch.nonzero(torch.abs(unstable_pre_conc_lb) <= torch.abs(unstable_pre_conc_ub)).squeeze() 
            mostly_active = (unstable_relus_idx[0][mostly_active],unstable_relus_idx[1][mostly_active])
            # mostly_active = unstable_relus_idx[mostly_active]
            a = x.max_lb[mostly_active] /  (x.max_lb[mostly_active] - x.conc_lb[mostly_active])
            a[x.max_lb[mostly_active] < 0] = 0.
            if(len(a.shape) > 0):
                a = a.unsqueeze(1)
            post_interval.l[mostly_active] = a * x.l[mostly_active]
            
            # post_interval.l[mostly_active] *= 0

            #Upper bound approximation
            unstable_pre_min_ub = x.min_ub[unstable_relus_idx]
            zero_crossing = torch.nonzero(unstable_pre_min_ub <= 0).squeeze()
            zero_crossing = (unstable_relus_idx[0][zero_crossing],unstable_relus_idx[1][zero_crossing])
            # zero_crossing = unstable_relus_idx[zero_crossing]
            a = x.conc_ub[zero_crossing] / (x.conc_ub[zero_crossing] - x.min_ub[zero_crossing])
            if(len(a.shape) > 0):
                a = a.unsqueeze(1)
            post_interval.u[zero_crossing] = a *  x.u[zero_crossing]
            post_interval.u[...,-1][zero_crossing] -= a.squeeze() * x.min_ub[zero_crossing]

        #Handle fixed relus
        if(layer_mask):
            for relu_idx, phase in layer_mask:
                if(phase == 0):
                    post_interval.l[:,relu_idx] = 0
                    post_interval.u[:,relu_idx] = 0
                elif(phase == 1):
                    post_interval.l[:,relu_idx] = self.pre_symbolic.u[:,relu_idx]
                    post_interval.u[:,relu_idx] = self.pre_symbolic.u[:,relu_idx]

        post_interval.concretize()
        torch.nn.functional.relu(post_interval.conc_bounds, inplace=True)
        self.post_symbolic = post_interval
        return post_interval

    @property
    def pre_conc_bounds(self):
        bounds = torch.concat((self.pre_conc_lb, self.pre_conc_ub), dim = 0).T
        return bounds
    
    @property 
    def post_conc_bounds(self):
        bounds = torch.concat((self.post_conc_lb, self.post_conc_ub), dim = 0).T
        return bounds

    @property 
    def post_conc_lb(self):
        # lb = torch.max(torch.zeros_like(self.post_symbolic.conc_lb), self.post_symbolic.conc_lb)
        return self.post_symbolic.conc_lb

    @property 
    def post_conc_ub(self):
        return self.post_symbolic.conc_ub

    @property 
    def pre_conc_lb(self):
        return self.pre_symbolic.conc_lb

    @property 
    def pre_conc_ub(self):
        return self.pre_symbolic.conc_ub