####################################################################################
# import libraries

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import normal, uniform, multivariate_normal, bernoulli, multinomial, chi2
from torch.autograd import Variable
from torch.nn import functional as F
import copy
import os # to check if file exists
from shutil import copyfile # to copy backup file

####################################################################################
# Find GPU with most available memory
# from https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:"+str(get_freer_gpu()))
else:
    device = torch.device("cpu")

####################################################################################
# unsqueeze at the specified dimension and duplicate the tensor the specified number of times along this new dimension
# tensor is cloned to avoid potential issues down the line
# if is negative, then -1 adds a new final dimension, -2 adds one before current final dimension, etc.
def dup_along_newdim(a,dim,num_rep,clone=False):
    if dim<0:
        dim += (a.dim()+1)
    if clone:
        return a.unsqueeze(dim).expand(dim*[-1] + [num_rep] + (a.dim()-dim)*[-1]).clone()
    else:
        return a.unsqueeze(dim).expand(dim*[-1] + [num_rep] + (a.dim()-dim)*[-1])

####################################################################################
# batch diagonal matrix (last coordinate of mat defines a diagonal matrix for each batch)
# https://github.com/pytorch/pytorch/issues/12160
def batch_diag(input):
    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N) 
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims,device=device)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal 
    output.as_strided(input.size(), strides ).copy_(input)
    return output    

# implementation of sampling from Wishart distribution
class Wishart:
    def __init__(self,V,df):
        self.V = V
        self.df = df
        
        self.p = V.size()[0]
        self.L = torch.cholesky(V,upper=False)
        self.chisq = chi2.Chi2(torch.as_tensor([self.df - i for i in range(self.p)],dtype=torch.get_default_dtype(),device=device))
    
    # if cholesky is True, returns the cholesky decomposition of the output
    def sample(self,nlist,cholesky=False):
        A = torch.tril(torch.randn(nlist + 2*[self.p],device=device),diagonal=-1) + batch_diag(torch.sqrt(self.chisq.sample(nlist)))
        A = torch.matmul(self.L,A)
        if cholesky:
            return A
        else:
            return torch.matmul(A,A.transpose(-1,-2))

####################################################################################
# Randomly shuffle the third dimension of an nbatch x n x dimW array
# Use a different random shuffle for each batch
def shuffle_dim3(a):
    pp = [torch.randperm(a.size()[2],device=device) for i in range(a.size()[0])]
    inds = torch.stack(pp,0).unsqueeze(1).repeat(1,a.size()[1],1)
    return torch.gather(a, 2, inds)

# Randomly shuffle the third dimension of a nbatch x n array
# Use a different random shuffle for each batch
def shuffle_dim2(a):
    pp = [torch.randperm(a.size()[1],device=device) for i in range(a.size()[0])]
    inds = torch.stack(pp,0)
    return torch.gather(a, 1, inds)


####################################################################################
# mean center and standardize a tensor along a prespecified dimension

def standardize(x,dim=None,eps=0):
    if dim is None:
        return (x - x.mean())/(x.std()+eps)
    else:
        return (x-x.mean(dim=dim,keepdim=True).expand_as(x))/(x.std(dim=dim,keepdim=True).expand_as(x)+eps)

####################################################################################
# Generate all power sets of the input
# https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
# returns an iterable of all power sets
def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    out = []
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
        
# list(powerset([1,2,3]))

####################################################################################
# create ReLU_net module
class ReLU_net(nn.Module):
    def __init__(self, input_size, num_hidden, hidden_size, output_size, leaky=False):
        super(ReLU_net, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size).to(device)])
        if num_hidden>1:
            for _ in range(num_hidden-1):
                self.layers.append(nn.Linear(hidden_size, hidden_size).to(device))
        self.layers = self.layers.append(nn.Linear(hidden_size, output_size).to(device))
                
        if leaky:
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i<(len(self.layers)-1):
                x = self.activation(x)
        return x
    
####################################################################################
# Equivariant network

# dims is the set of dims over which to be equivariant to permutations (should be negative numbers, and should all be less than -1)
# In the case that dims contains one value, each layer is a  multi-input-output channel equivariant layer as described in Zaheer et al. (2017)
# In the case tht dims contains two values, each layer is a multi-input-output channel equivariant layer as described in Hartford et al. (2018)
class equivar_layer(nn.Module):
    def __init__(self, input_size, output_size, dims, activation=None, pool="mean", bias_init = None):
        super(equivar_layer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(2**len(dims)):
            # (only include bias for first layer)
            self.layers.append(nn.Linear(input_size, output_size,bias=(True if i==0 else False)).to(device))
            if bias_init is not None and (i==0):
                nn.init.constant_(self.layers[i].bias.data, bias_init)

        # sort the dimensions (important for the order we take the sums in for the forward pass)
        dims.sort()
        self.dims = dims
        self.dim_combos = list(powerset(self.dims))
        
        if pool in ["max","mean"]:
            self.pool = pool
        else:
            raise Exception("Invalid pool.")
            
        if activation is None:
            self.activation = lambda x: x
        else:
            if activation=="leaky_relu":
                self.activation = nn.LeakyReLU()
            elif activation=="relu":
                self.activation = nn.ReLU()
            else:
                raise Exception("Invalid activation.")

    def forward(self, x):
        z = 0
        for i in range(len(self.dim_combos)):
            curr_dims = self.dim_combos[i]
            tmp = self.layers[i](x)

            if len(curr_dims)>0:
                if self.pool=="mean":
                    tmp = tmp.mean(dim=tuple(curr_dims),keepdim=True).expand_as(tmp)
                elif self.pool=="max":
                    tmp = tmp.max(dim=tuple(curr_dims),keepdim=True)[0].expand_as(tmp)
            z += tmp
        return self.activation(z)
    
# create equivar_net module
# dims is the set of dims to be equivariant over (should be negative numbers, and should all be less than -1)
# activation is to applied to all layers except the last one
class equivar_net(nn.Module):
    def __init__(self, input_size, num_hidden, hidden_size, output_size, dims, activation=None, pool="mean", bias_init=None):
        super(equivar_net, self).__init__()
        self.layers = nn.ModuleList([equivar_layer(input_size, hidden_size, dims, activation=activation, bias_init=bias_init).to(device)])
        if num_hidden>1:
            for _ in range(num_hidden-1):
                self.layers.append(equivar_layer(hidden_size, hidden_size, dims, activation=activation, pool=pool, bias_init=bias_init).to(device))
        self.layers = self.layers.append(equivar_layer(hidden_size, output_size, dims, activation=None, pool=pool, bias_init=bias_init).to(device))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
    
####################################################################################
## Prior for the covariates
class PriorW(nn.Module):
    def __init__(self, hidden_size, wdim):
        super(PriorW, self).__init__()
                
        self.wishart = Wishart(torch.diag(torch.ones(wdim,device=device)),2*wdim)
                
        self.wdim = wdim

    def forward(self, nbatch, n):
        A = torch.inverse(self.wishart.sample([nbatch],cholesky=True)).transpose(-2,-1)
        diag = torch.diag_embed((torch.diagonal(torch.matmul(A,A.transpose(-2,-1)),dim1=-2,dim2=-1))**(-0.5),dim1=-2,dim2=-1)

        # generate w
        w = torch.matmul(dup_along_newdim(torch.matmul(diag,A),1,n),torch.randn(nbatch,n,self.wdim,1,device=device)).squeeze(-1)

        return w

####################################################################################
# create Prior neural network class
# Linear regression

class PriorLinear(nn.Module):
    # If permute is set to true, then Pi will permute the beta outputs
    def __init__(self, hidden_size, num_hidden, wdim, s, permute=True):
        super(PriorLinear, self).__init__()
        
        self.permute = permute
        self.s = s
        self.wdim = wdim

        # magnitude of ell_1 norm of betas
        self.radius = 5
        
        # relu generator network for betas (independent output for each beta)
        if s==1: # use relu_net if s=1
            self.rn = ReLU_net(s, num_hidden, hidden_size, s, leaky=True)
        else: # otherwise using an equivar_net
            self.rn = equivar_net(1, num_hidden, hidden_size, 1, [-2], activation="leaky_relu", pool="mean")

        self.noise_sampler = normal.Normal(loc = torch.zeros(1,device=device), scale = torch.ones(1,device=device))
        
        # homoscedastic error distribution
        self.E = normal.Normal(loc = torch.zeros(1,device=device), scale = torch.ones(1,device=device))
        
        # W generator
        self.PiW = PriorW(hidden_size, wdim)

        # bernoulli with probability of success 1/2
        self.bern5 = bernoulli.Bernoulli(0.5)

        # softmax along dimension 1
        self.softmax = nn.Softmax(dim=1)

        self.tanh = nn.Tanh()

    def forward(self, nbatch, n, n_tilde=1):
        s = self.s
        wdim = self.wdim

        # sample w
        w_all = self.PiW(nbatch, n+n_tilde)

        # nbatch x s output
        if s==1:
            u = self.noise_sampler.sample([nbatch])
            beta = self.rn(u)
            # transform beta so that it falls in +/- self.radius and flip its sign at random
            beta = self.radius * self.tanh(beta) * (2*self.bern5.sample([nbatch,s]).to(device)-1)
        else:
            u = self.noise_sampler.sample([nbatch,s])
            beta = self.rn(u).squeeze(-1)
            # scale to surface of unit ball and then scale to have radius Unif(0,self.radius)
            beta = self.radius * (2*torch.rand(nbatch,s,device=device)-1) * self.softmax(beta)

        # append zeros if s<wdim
        if s<wdim:
            # nbatch x (wdim + 1) output
            beta = torch.cat((beta,torch.zeros(nbatch,wdim-s,device=device)),dim=1)
        
        if self.permute:
            # randomly permute indices (relevant if T not invariant to permutations of predictors)
            beta = shuffle_dim2(beta)

        # w_all is nbatch x n x wdim
        # beta is nbatch x wdim
        out = torch.matmul(w_all,beta.unsqueeze(-1))
        
        # generate y
        y = out[:,range(n),:] + self.E.sample([nbatch,n])
        
        return w_all[:,range(-n_tilde,0),:], w_all[:,range(n),:], y, out

####################################################################################
# create Prior neural network class
# GAM where each component is of bounded variation

class PriorGam(nn.Module):
    # If permute is set to true, then Pi will permute the beta outputs
    def __init__(self, udim, hidden_size, wdim, s, M=10, permute=True, num_hidden=4, n_fixed = 500):
        super(PriorGam, self).__init__()
        
        self.permute = permute
        self.s = s
        self.M = M
        self.wdim = wdim
        
        # relu generator network
        self.rn = ReLU_net(udim, num_hidden, hidden_size, s, leaky=True)
        # source of noise
        self.noise_sampler = normal.Normal(loc = torch.zeros(udim,device=device), scale = torch.ones(udim,device=device))

        # homoscedastic error distribution
        self.E = normal.Normal(loc = torch.zeros(1,device=device), scale = torch.ones(1,device=device))
        
        # W generator
        self.PiW = PriorW(hidden_size, wdim)
        
        # bernoulli with probability of success 1/2
        self.bern5 = bernoulli.Bernoulli(0.5)

        # number of fixed Ws used to define jump locations
        self.n_fixed = n_fixed

    def forward(self, nbatch, n, n_tilde=1):
        # number of points to use when generating the jumps
        n_fixed = self.n_fixed
        n_obs = n+n_tilde
        # sample w
        w_all = self.PiW(nbatch, n_fixed + n_obs)
                
        ndim = w_all.dim()
        s = self.s
        
        if self.permute:
            # randomly permute indices (important when s<wdim, unless T perm invariant in columns)
            w_all = shuffle_dim3(w_all)
        
        w_fixed = w_all[:,range(n_fixed),:]
        w_obs = w_all[:,range(-(n_obs),0),:]
        
        # order and rank of the values (by coordinate of w_fixed)
        order_inds_w_fixed = torch.argsort(w_fixed[:,:,range(s)].permute(0,2,1),dim=2)
        rank_inds_w_fixed = torch.argsort(order_inds_w_fixed)
        
        # order and rank of the values (by coordinate of w_all)
        order_inds = torch.argsort(w_all[:,:,range(s)].permute(0,2,1),dim=2)
        rank_inds = torch.argsort(order_inds)
        
        # nbatch x n_fixed x s output
        one_d_funs = self.rn(self.noise_sampler.sample([nbatch,n_fixed]))

        # Flip sign of increments at random
        one_d_funs = one_d_funs * (2*self.bern5.sample([nbatch,n_fixed,s]).to(device)-1)
        
        # define variation norm
        var_norm = torch.abs(one_d_funs).sum(dim=2).sum(dim=1)
        # add a small number to var_norm to avoid division by zero later
        var_norm = var_norm + 0.01

        one_d_funs = (self.M * one_d_funs / dup_along_newdim(dup_along_newdim(var_norm,-1,n_fixed),-1,s))

        one_d_funs = torch.gather(one_d_funs.permute(0,2,1),2,rank_inds_w_fixed).permute(0,2,1)
        one_d_funs = torch.cat((one_d_funs,torch.zeros(nbatch,n_obs,s,device=device)),dim=-2)
        one_d_funs = torch.gather(one_d_funs.permute(0,2,1),2,order_inds).permute(0,2,1)

        # Take the cumulative sum of the increments to get the regression function
        one_d_funs = torch.cumsum(one_d_funs,dim=1)

        # Reorder the one_d_funs to correspond to the ranks of the coordinates of ws
        one_d_funs = torch.gather(one_d_funs.permute(0,2,1),2,rank_inds).permute(0,2,1)
        
        # for each observation, sum out the one_d_funs across the coordinates of w
        one_d_funs = one_d_funs.sum(dim=2).unsqueeze(-1)

        # generate y
        y = one_d_funs[:,range(-n_obs,-n_tilde),:] + self.E.sample([nbatch,n])
    
        return w_obs[:,range(-n_tilde,0),:], w_obs[:,range(-n_obs,-n_tilde),:], y, one_d_funs[:,range(-n_obs,0),:]

####################################################################################
# Class of equivariant estimators used when ablation==0 (all experiments except one)

class Predictor(nn.Module):
    def __init__(self, hidden_size1, num_hidden1, pool_size1, hidden_size2, num_hidden2, pool_size2, hidden_size3, num_hidden3, pool_size3, hidden_size4, num_hidden4, output_size, rank_based = False):
        super(Predictor, self).__init__()
        
        self.np_net = equivar_net(2, num_hidden1, hidden_size1, pool_size1, [-3,-2], activation="leaky_relu", pool="mean")
        self.p_net1 = equivar_net(pool_size1, num_hidden2, hidden_size2, pool_size2, [-2], activation="leaky_relu", pool="mean")
        self.p_net2 = equivar_net(pool_size2+1, num_hidden3, hidden_size3, pool_size3, [-2], activation="leaky_relu", pool="mean")
        self.out_net = ReLU_net(pool_size3, num_hidden4, hidden_size4, 1, leaky=True)

        self.leakyrelu = nn.LeakyReLU()
        
        self.rank_based = rank_based
        
    def forward(self, w_tilde, w, y): 

        if w.dim()==2:
            w = w.unsqueeze(0) 
            y = y.unsqueeze(0)

        nbatch, n, wdim = w.size()

        orig_w_tilde_dim = w_tilde.dim()
        if w_tilde.dim()==2:
            w_tilde = w_tilde.unsqueeze(0)
        n_tilde = w_tilde.size()[1]

        # Replace w and w_tilde by ranks
        if self.rank_based:
            # compute ranks of w_tilde in w
            # note: runs in quadratic time for a given batch/coordinate of w.
            #       could be run in linear time by presorting w_tilde and w, but then not clear to me how to distribute this.
            #       (so, the quadratic-time version here actually runs faster on a GPU for reasonably-sized w_tilde)
            # nbatch x n_tilde x n x p
            w_tilde = torch.as_tensor((dup_along_newdim(w_tilde,-2,n) > dup_along_newdim(w,-3,n_tilde)),dtype=torch.get_default_dtype(),device=device).sum(dim=-2)

            # replace w by ranks
            w = torch.as_tensor(torch.argsort(torch.argsort(w,dim=1),dim=1),dtype=torch.get_default_dtype(),device=device)

        mu = y.mean(dim=1)
        ctr_train = dup_along_newdim(mu,1,n)
        ctr_test = dup_along_newdim(mu,1,n_tilde).squeeze(-1)

        # center and standardize y (will unstandardize output at the end)
        sigma = y.std(dim=1)
        y = (y-ctr_train)/dup_along_newdim(sigma,1,n)
      
        # standardize w and w_tilde
        if self.rank_based:
            mu_w = (n-1)/2
            sigma_w = np.sqrt((n-1)*(2*n-1)/6-((n-1)/2)**2)
            w = (w - mu_w)/sigma_w
            w_tilde = (w_tilde - mu_w)/sigma_w
        else:
            mu_w = w.mean(dim=-2)
            sigma_w = w.std(dim=-2)
            w = (w - mu_w.unsqueeze(-2).expand_as(w))/sigma_w.unsqueeze(-2).expand_as(w)
            w_tilde = (w_tilde - mu_w.unsqueeze(-2).expand_as(w_tilde))/sigma_w.unsqueeze(-2).expand_as(w_tilde)

        # first compute "sufficient statistics" from the data (don't use w_tilde)
        x = torch.cat((w.unsqueeze(3),dup_along_newdim(y,dim=2,num_rep=wdim)),dim=3)
    
        x = self.np_net(x).mean(dim=-3)

        # process sufficient statistcs a bit more before appending the w_tilde at which to make a prediction
        x = self.leakyrelu(self.p_net1(x))

        # Concatenate nbatch x n_tilde x p x d (the x tensor is duplicated across dim 2) by nbatch x n_tilde x p x 1 (unsqueezed w_tilde)
        x = torch.cat((dup_along_newdim(x,dim=1,num_rep=n_tilde),w_tilde.unsqueeze(-1)),dim=-1)
        x = self.p_net2(x).mean(dim=-2)
        x = self.out_net(x).squeeze(-1)

        # unstandardize prediction
        x = x * sigma.expand_as(x) + ctr_test
            
        if orig_w_tilde_dim==0:
            x = x.squeeze(0)

        return x


####################################################################################
# ablation==1: Not invariant to permutations of the n observations

class PredictorNoN(nn.Module):
    def __init__(self, hidden_size1, num_hidden1, pool_size1, hidden_size2, num_hidden2, pool_size2, hidden_size3, num_hidden3, pool_size3, hidden_size4, num_hidden4, output_size, rank_based = False,n=100):
        super(PredictorNoN, self).__init__()
        
        self.np_net = equivar_net(2*n, num_hidden1, hidden_size1, pool_size1, [-2], activation="leaky_relu", pool="mean")
        self.p_net1 = equivar_net(pool_size1, num_hidden2, hidden_size2, pool_size2, [-2], activation="leaky_relu", pool="mean")
        self.p_net2 = equivar_net(pool_size2+1, num_hidden3, hidden_size3, pool_size3, [-2], activation="leaky_relu", pool="mean")
        self.out_net = ReLU_net(pool_size3, num_hidden4, hidden_size4, 1, leaky=True)

        self.leakyrelu = nn.LeakyReLU()
        
        self.rank_based = rank_based
        
    def forward(self, w_tilde, w, y): # w_tilde is the feature/set of features at which to make a predictions, and (w,y) is the dataset to train on
        if w.dim()==2:
            w = w.unsqueeze(0) 
            y = y.unsqueeze(0)

        nbatch, n, wdim = w.size()

        orig_w_tilde_dim = w_tilde.dim()
        if w_tilde.dim()==2:
            w_tilde = w_tilde.unsqueeze(0)
        n_tilde = w_tilde.size()[1]

        # Replace w and w_tilde by ranks
        if self.rank_based:
            # compute ranks of w_tilde in w
            # note: runs in quadratic time for a given batch/coordinate of w.
            #       could be run in linear time by presorting w_tilde and w, but then not clear to me how to distribute this.
            #       (so, the quadratic-time version here actually runs faster on a GPU for reasonably-sized w_tilde).
            #       perhaps someone else can improve this implementation to make this run more quickly.
            # nbatch x n_tilde x n x p
            w_tilde = torch.as_tensor((dup_along_newdim(w_tilde,-2,n) > dup_along_newdim(w,-3,n_tilde)),dtype=torch.get_default_dtype(),device=device).sum(dim=-2)

            # replace w by ranks
            w = torch.as_tensor(torch.argsort(torch.argsort(w,dim=1),dim=1),dtype=torch.get_default_dtype(),device=device)

        mu = y.mean(dim=1)
        ctr_train = dup_along_newdim(mu,1,n)
        ctr_test = dup_along_newdim(mu,1,n_tilde).squeeze(-1)

        # center and standardize y (will unstandardize output at the end)
        sigma = y.std(dim=1)
        y = (y-ctr_train)/dup_along_newdim(sigma,1,n)
      
        # standardize w and w_tilde
        if self.rank_based:
            mu_w = (n-1)/2
            sigma_w = np.sqrt((n-1)*(2*n-1)/6-((n-1)/2)**2)
            w = (w - mu_w)/sigma_w
            w_tilde = (w_tilde - mu_w)/sigma_w
        else:
            mu_w = w.mean(dim=-2)
            sigma_w = w.std(dim=-2)
            w = (w - mu_w.unsqueeze(-2).expand_as(w))/sigma_w.unsqueeze(-2).expand_as(w)
            w_tilde = (w_tilde - mu_w.unsqueeze(-2).expand_as(w_tilde))/sigma_w.unsqueeze(-2).expand_as(w_tilde)

        # first compute "sufficient statistics" from the data (don't use w_tilde)
        # nbatch x p x 2*n output (n for the corresponding predictor, n for the outcomes)
        x = torch.cat((w.permute(0,2,1),dup_along_newdim(y.squeeze(-1),dim=1,num_rep=wdim)),dim=-1)

        x = self.np_net(x)

        # process sufficient statistcs a bit more before appending the w_tilde at which to make a prediction
        x = self.leakyrelu(self.p_net1(x))

        # Concatenate nbatch x n_tilde x p x d (the x tensor is duplicated across dim 2) by nbatch x n_tilde x p x 1 (unsqueezed w_tilde)
        x = torch.cat((dup_along_newdim(x,dim=1,num_rep=n_tilde),w_tilde.unsqueeze(-1)),dim=-1)
        x = self.p_net2(x).mean(dim=-2)
        x = self.out_net(x).squeeze(-1)

        # unstandardize prediction
        x = x * sigma.expand_as(x) + ctr_test
            
        if orig_w_tilde_dim==0:
            x = x.squeeze(0)

        return x

####################################################################################
# Initialize Prior
# (M and udim are only relevant if gam=True)
def initPi(s,udim,wdim,M=None,gam=False,lr=0.001):
    if gam:
        Pi = PriorGam(udim = udim,
            hidden_size = 40,
            wdim = wdim,
            s=s,
            M=M,
            num_hidden = 4,
            permute = False).to(device)
    else:
        Pi = PriorLinear(
            hidden_size = 40,
            wdim = wdim,
            s=s,
            num_hidden = 4,
            permute = False).to(device)  # caution: only setting to False because T is equivariant to permutations of the predictors
    
    Pi_opt = optim.Adam(Pi.parameters(),
        lr = lr, 
        betas = (0, 0.999), 
        eps = 1e-08)
    Pi_lr_lambda = lambda epoch: max(1.0,epoch)**(-0.25)
    Pi_sched = torch.optim.lr_scheduler.LambdaLR(Pi_opt, Pi_lr_lambda, last_epoch=-1)
    
    return Pi, Pi_opt, Pi_sched

####################################################################################
# Initialize Predictor

def initT(lr=0.001,rank_based=False,gam=False,ablation=0):
    if ablation==0:
        T = Predictor(hidden_size1 = 100,
                         num_hidden1 = 10,
                         pool_size1 = 50,
                         hidden_size2 = 100,
                         num_hidden2 = 3,
                         pool_size2 = 50,
                         hidden_size3 = 100,
                         num_hidden3 = 10,
                         pool_size3 = 10,
                         hidden_size4 = 100,
                         num_hidden4 = 3,
                         output_size = 1,
                         rank_based = rank_based).to(device)
    elif ablation==1: # T not invariant to permutations of predictors
        T = PredictorNoN(hidden_size1 = 100,
                         num_hidden1 = 10,
                         pool_size1 = 50,
                         hidden_size2 = 100,
                         num_hidden2 = 3,
                         pool_size2 = 50,
                         hidden_size3 = 100,
                         num_hidden3 = 10,
                         pool_size3 = 10,
                         hidden_size4 = 100,
                         num_hidden4 = 3,
                         output_size = 1,
                         rank_based = rank_based).to(device)


    T_opt = optim.Adam(T.parameters(), 
                             lr = lr, 
                             betas = (0.25, 0.999), 
                             eps = 1e-08)
    T_lr_lambda = lambda epoch: max(1.0,epoch)**(-0.15)
    T_sched = torch.optim.lr_scheduler.LambdaLR(T_opt, T_lr_lambda, last_epoch=-1)
    
    return T, T_opt, T_sched

####################################################################################
# define loss function
# CAUTION: gradients don't propagate through target!
MSE = nn.MSELoss()

# gradients propagate through target
def MSE2(est,targ):
    return ((est-targ)**2).mean()
####################################################################################
# Load a trained estimator and prior
def load_model(T, T_opt, T_sched, Pi, Pi_opt, Pi_sched, fl, fl_backup = None, device=device):
    """load all estimators + optimizers to resume training from prev checkpoint"""
    if fl_backup is not None:
        try:
            checkpoint = torch.load(fl,map_location=device)
        except:
            copyfile(fl_backup,fl) # replace fl by fl_backup
            checkpoint = torch.load(fl,map_location=device)
    else:
        checkpoint = torch.load(fl,map_location=device)
        
    start_iter = (checkpoint['iteration'])+1
    T.load_state_dict(checkpoint['T_state_dict'])
    T_opt.load_state_dict(checkpoint['T_opt_state_dict'])
    T_sched.load_state_dict(checkpoint['T_sched_state_dict'])
    Pi.load_state_dict(checkpoint['Pi_state_dict'])
    Pi_opt.load_state_dict(checkpoint['Pi_opt_state_dict'])
    Pi_sched.load_state_dict(checkpoint['Pi_sched_state_dict'])
    loss_list = checkpoint['loss_list']

    return start_iter, loss_list
####################################################################################
# Evaluate performance of T against a prior Pi
def interrogate(T,Pi,n,n_tilde,numbatch=100,batchsize=1000):
    risks = torch.zeros(numbatch)
    
    for i in range(numbatch):
        # conditional mean from prior
        with torch.no_grad():
            w_tilde, w, y, regfun = Pi(batchsize,n,n_tilde=n_tilde)

        # evaluate T
        T_out = T(w_tilde, w, y).detach()

        # compute risk of T
        risks[i] = MSE(T_out, regfun[:,range(-n_tilde,0),:].squeeze(-1))
    
    risk = risks.mean()
    se = (risks.var()/numbatch).sqrt()
    
    return risk, se
####################################################################################
# raise exception because T loss became nan since last save
class nanError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
####################################################################################
# adversarial meta-training

# if stopped in middle of writing to fl (so that get EOF error), then network can be recovered from fl_backup
def train(niter,n,n_tilde,nbatch,T,T_opt,T_sched,Pi,Pi_opt,Pi_sched,fl,loadm=False,new_T_lr=None,new_Pi_lr=None,fixedPi=False,nT=1,firstPiIter=0):
    # name of backup file (can be used to recover network if train is stopped in the middle of writing to file,
    #  leading to an EOF error)
    fl_backup = os.path.splitext(fl)[0] + "_backup" + os.path.splitext(fl)[1]
    
    if loadm and os.path.isfile(fl):
        iteration, loss_list = load_model(T, T_opt, T_sched, Pi, Pi_opt, Pi_sched, fl, fl_backup = fl_backup)
        print("Resuming training at iteration ",iteration,".",flush=True)
    else:
        if loadm:
            print("Load file did not exist. Starting at iteration 0.",flush=True)
        if os.path.isfile(fl) and not loadm:
            raise Exception("Specified loadm=False, but file already exists. Please manually delete "+fl+" if you wish to train a new model.")
        iteration = 0
        loss_list = []
    
    # Set new learning rates if requested
    if new_Pi_lr is not None:
        for g in Pi_opt.param_groups:
            print("Setting Pi optimizer learning rate to "+str(new_Pi_lr),flush=True)
            g['lr'] = new_Pi_lr
    if new_T_lr is not None:
        for g in T_opt.param_groups:
            print("Setting T optimizer learning rate to "+str(new_T_lr),flush=True)
            g['lr'] = new_T_lr
        
    while iteration<=niter:
        loss_list0 = []
    
        for _ in range(nT):
        
            # zero out previous gradients
            T.zero_grad()

            # gradient step on T network

            # sample from Pi
            with torch.no_grad():
                w_tilde, w, y, regfun = Pi(nbatch,n,n_tilde=n_tilde)

            # get estimate of mu(x) from procedure network
            T_out = T(w_tilde, w, y)
            T_out2 = T(w_tilde, w, -y)

            # get loss and gradient
            T_loss = MSE(T_out, regfun[:,range(-n_tilde,0),:].squeeze(-1)) + MSE(-T_out2, regfun[:,range(-n_tilde,0),:].squeeze(-1))

            T_loss.backward()
                    
            # perform one gradient descent step
            T_opt.step()
            
            T_sched.step()

            # append values to list
            loss_list0.append(T_loss.item())        

        # append avg values to list
        loss_list.append(np.mean(loss_list0))
        
        if (not fixedPi) & (iteration>=firstPiIter): # start training Pi if Pi is not fixed and the firstPiIter*nT iterations on T have already occurred
            # gradient step on Pi network

            # zero out previous gradients
            Pi.zero_grad()

            # sample from Pi
            w_tilde, w, y, regfun = Pi(nbatch,n,n_tilde=n_tilde)

            # get estimate of mu(x) from procedure network
            T_out = T(w_tilde, w, y)

            # get loss and gradient
            Pi_loss = torch.neg(MSE2(T_out, regfun[:,range(-n_tilde,0),:].squeeze(-1)))

            Pi_loss.backward()

            # perform one gradient descent step
            Pi_opt.step()
                        
            
            Pi_opt.step()

            Pi_sched.step()
        
            if np.isnan(Pi_loss.item()): # check if Pi_loss is nan (if it is, stop, otherwise, save)
                raise nanError("nan occurred since last save. New values of T and Pi have not been saved so that the old values can still be recovered from file.")

        # print diagnostics
        if (iteration % 500 == 0) or (iteration==niter):
            print('Iteration ',
                '{:6.0f}'.format(iteration),
                ':',
                ' ' * 2,
                'Loss-adversarial = ',
                '{:7.4f}'.format(T_loss.item()),flush=True)

            print("...saving a checkpoint at iter: ", iteration,flush=True)
            # first copy old fl to backup (assuming it exists)
            if os.path.exists(fl):
                copyfile(fl, fl_backup)
                        
            # save network to fl
            torch.save({
                'iteration': iteration,
                'T_state_dict': T.state_dict(),
                'Pi_state_dict': Pi.state_dict(),
                'T_opt_state_dict': T_opt.state_dict(),
                'Pi_opt_state_dict': Pi_opt.state_dict(),
                'T_sched_state_dict': T_sched.state_dict(),
                'Pi_sched_state_dict': Pi_sched.state_dict(),
                'loss_list': loss_list
            }, fl)
            
            # remove backup (assuming it exists)
            if os.path.exists(fl_backup):
                os.remove(fl_backup)
            
        iteration+=1
        
    return T, Pi, loss_list
####################################################################################
