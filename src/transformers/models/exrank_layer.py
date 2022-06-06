# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.



#TODO(yhq): 1)output the gammar value[the argument for vis_tools.py] to check the function form. 2)
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from .soft_expand import soft_exponential
# from .soft_expand_beta import soft_exponential_beta
from .soft_transform import soft_yhq
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as pltno
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
# from .whitebert_utils import whitening_torch_final,whitebert_debug
# def spectral_norm(w, r=5):
#     w_shape = torch.shape(w)
#     in_dim = np.prod(w_shape[:-1]).astype(int)
#     out_dim = w_shape[-1]
#     w = torch.reshape(w, (in_dim, out_dim))
#     u = torch.ones((1, in_dim))
#     for i in range(r):
#         v = torch.l2_normalize(torch.dot(u, w))
#         u = torch.l2_normalize(torch.dot(v, torch.transpose(w)))
#     return torch.sum(K.dot(torch.dot(u, w), torch.transpose(v)))

#TODO(yhq): observe the token uniformity before and after rescaling the eigenvalue distribution. NEED to add token tags/word
def apply_mask(x,p):
    r = F.gumbel_softmax(p,hard=True,dim=2)[:,:,1:2]
    x_prime = r * x
    return x_prime

def vis_tokenUni(tokens,picname,ifpca):
    # time_start = time.time()
    batch_id = 0
    tokens = tokens[batch_id].cpu().detach().numpy()
    if ifpca:
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(tokens)
        tokens = pca_result_50
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tokens)
    d = {'tsne-2d-one': tsne_results[:,0], 'tsne-2d-two':tsne_results[:,1]}
    df_subset = pd.DataFrame(data=d)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    # palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=None,
    )
    plt.savefig(picname)

class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True,llayer = False):
        super(LayerNormImpl, self).__init__()
        self.norm_mode = args.lnv
        # self.sigma = args.sigma
        #TODO(yhq0513): dim=768, hidden_dim=4*dim
        #TODO(byj):
        self.hidden = 768
        self.decay_alpha = -0.2
        self.ifmask = False
        self.llayer = False
        self.soft_exp = soft_exponential(in_features=args.max_length,alpha=None,hidden_dim=self.hidden,decay_alpha=self.decay_alpha,ifmask=self.ifmask)
        # self.soft_yhq = soft_yhq(in_features=args.max_length,alpha=None,hidden_dim=self.hidden,decay_alpha=self.decay_alpha,ifmask=self.ifmask)      
        # self.soft_exp_beta = soft_exponential_beta(in_features=args.max_length,alpha=None,beta=None,hidden_dim=self.hidden,decay_alpha=self.decay_alpha,ifmask=self.ifmask)
        if args.spectral_norm == True:
            # print("use spec")
            self.raw_exrank = nn.Linear(self.hidden,self.hidden)
            self.exrank_linear = torch.nn.utils.spectral_norm(self.raw_exrank)
        else:
            self.exrank_linear = nn.Linear(hidden,hidden)
        if args.exrank_nonlinear == 'relu':
            self.exrank_nonlinear = nn.ReLU()
        elif args.exrank_nonlinear == 'selu':
            self.exrank_nonlinear = nn.SELU()
        elif args.exrank_nonlinear == "elu":
            self.exrank_nonlinear = nn.ELU()
    
        self.rescale_weight = nn.Parameter(torch.Tensor(args.max_seq_length,hidden))
        #TODO(yhq): init the gamma in different ways, should rela
        # self.log_base = nn.Parameter(torch.Tensor(1))
        self.logbase = Variable(torch.ones(1), requires_grad=True).cuda()
        # self.logbase = Variable(torch.ones(8,1), requires_grad=True).cuda() 
        # self.logbase = torch.tensor(20.0, requires_grad=True)
        # self.exrank_bias = nn.Parameter(torch.Tensor(args.max_seq_length,hidden))

        if self.norm_mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_parameter('rescale_weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

            nn.init.ones_(self.rescale_weight)
            # nn.init.zeros_()

    def forward(self, input):
        seed = np.random.choice(30,1)
        if seed[0] == 1:
            self.norm_mode = "soft_expand"
        else:
            self.norm_mode = "origin"
        if self.norm_mode == 'exrank_gx' or self.llayer: #apply regularization to g(x) weight
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input-mean)/(std+self.eps)
            output = input*self.exrank_nonlinear(self.exrank_linear(input_norm))
            gx_weight = self.exrank_linear.weight
            return (output,gx_weight)
        elif self.norm_mode == 'origin':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input-mean)/(std+self.eps)
            output = self.weight*input_norm+self.bias
            return (output,self.weight)
        elif self.norm_mode == 'no_norm':
            return (input,)
        elif self.norm_mode == 'average':
            u,s,v = torch.svd(input)
            newS = torch.mean(s,dim=-1).unsqueeze(-1)*torch.ones(s.shape).cuda()
            rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
            return (new_input,rescale_s_dia)
        elif self.norm_mode == 'linear':
            K = 1.1 
            u,s,v = torch.svd(input)
            maxS = torch.max(s,dim=1).values.unsqueeze(-1) #[8,128]
            newS = maxS - (maxS-s)/K
            rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
            return (new_input,rescale_s_dia)
        elif self.norm_mode == 'rescale':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input-mean)/(std+self.eps)
            output = self.rescale_weight*input_norm
            return (output,self.rescale_weight)
        #TODO(yhq:) add the "distribution shift" function:
        elif self.norm_mode == "shift":
            gamma = self.logbase #original
            #step1: svd the input
            u,s,v = torch.svd(input) #u[8,128,128] s[8,128] v[8,768,128]
            maxS = torch.max(s,dim=1).values.unsqueeze(-1) #[8,128]
            #TODO(yhq): 
            # rescale_s = torch.log10(s/maxS+torch.ones_like(s))/torch.log10(gamma).cuda() #without rescaling
            #TODO(yhq): 0506 GuiLin
            rescale_s = torch.log10(gamma*s/maxS+torch.ones_like(s))/torch.log10(torch.ones_like(s)+gamma).cuda()
            rescale_s_dia = torch.diag_embed(maxS*rescale_s,dim1=-2,dim2=-1)#[8,128,128]dia
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1)) #[8,128,128]
            return (new_input,gamma)
        elif self.norm_mode == "soft_expand":
            u,s,v = torch.svd(input)
            maxS = torch.max(s,dim=1).values.unsqueeze(-1)
            newS,alpha = self.soft_exp(input,s)#[8,128]
            #TODO(yhq0507): normalize the new_s to maxvalue=1
            maxNewS = torch.max(newS,dim=1).values.unsqueeze(-1)
             #make the maxS unchanged
            rescale_number = maxNewS/maxS
            newS = newS/rescale_number
            rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
            # print(alpha.item())
            return (new_input,alpha)
        elif self.norm_mode == "soft_transform":
            u,s,v = torch.svd(input)
            maxS = torch.max(s,dim=1).values.unsqueeze(-1)
            newS,alpha = self.soft_yhq(input,s)#[8,128]
            #TODO(yhq0507): normalize the new_s to maxvalue=1
            maxNewS = torch.max(newS,dim=1).values.unsqueeze(-1)
             #make the maxS unchanged
            rescale_number = maxNewS/maxS
            newS = newS/rescale_number
            rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
            # print(alpha.item())
            return (new_input,alpha)
        elif self.norm_mode == "e_decay": #042022 expontialDecay function 
            u,s,v = torch.svd(input) #s [bs,seq_len]
            rescale_s_dia = torch.diag_embed(s,dim1=-2,dim2=-1)
            new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
            #add regulariztion to s
            return (new_input,s)
        # elif self.norm_mode == "soft_expand_beta":
        #     u,s,v = torch.svd(input)
        #     maxS = torch.max(s,dim=1).values.unsqueeze(-1)
        #     newS,alpha = self.soft_exp_beta(input,s)#
        #     # if alpha.item()>-0.5 or alpha.item()<-0.5:
        #         # print(alpha.item())
        #     maxNewS = torch.max(newS,dim=1).values.unsqueeze(-1)
        #     rescale_number =  maxNewS/maxS #make the maxS unchanged
        #     newS = newS/rescale_number
        #     rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
        #     new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(2,1))
        #     return (new_input,alpha)
        else:
            return (input,)

def NormFuncs(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None):
    if args is not None:
        return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    #     if args.lnv != 'origin':
    #         return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    # if not export and torch.cuda.is_available():
    #     try:
    #         from apex.normalization import FusedLayerNorm
    #         return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    #     except ImportError:
    #         pass
    # return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)