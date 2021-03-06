#Translate and rescale the eigenvalue distribution. Use two parameters to control the translte and scale degree, respectively, based on  the soft_expand function.
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
class soft_exponential_beta(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha = None, beta=None, hidden_dim=768, mask_hidden_dim=300,ifmask=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential_beta,self).__init__()
        self.in_features = in_features
        self.ifmask = ifmask

        # initialize alpha
        # if alpha == None:
        self.alpha = torch.FloatTensor([-0.5]).cuda()
        self.beta = torch.FloatTensor([0.]).cuda()
        self.alpha.requires_grad = True
        self.beta.requires_grad = True
            # self.alpha = nn.Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        # else:
            # self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha

        # if beta == None:
            # self.beta = nn.Parameter(torch.tensor(0.0)) 
        # else:
            # self.beta = nn.Parameter(torch.tensor(beta)) # create a tensor out of alpha
            
        # self.alpha.requiresGrad = True # set requiresGrad to true!

        #TODO(yhq:0507) add parameters for mask function
        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.leaky_relu}
        self.activation = self.activations["relu"]
        self.linear_layer = nn.Linear(hidden_dim, mask_hidden_dim)
        self.hidden2p = nn.Linear(mask_hidden_dim, 2)

    def mask(self, x,s,fs):
        '''use mask to decide which eigenvalue should be kept unchanged as s or changed to fs!
        x: the semantic representation used for mask calculation;
        s: original eigenvalue distribution;
        fs: new eigenvalue distribution after applying soft_expand function'''
        temps = self.activation(self.linear_layer(x))
        p = self.hidden2p(temps)  # seqlen, bsz, dim
        mask = F.gumbel_softmax(p,hard=True,dim=2)
        newS = mask[:,:,0]*s+mask[:,:,1]*fs
		# x_prime = r * x
        return newS

    def forward(self, input,s):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        #TODO(yhq): beta control the translate factor.
        if (self.alpha == 0.0):
            fs = s
            if self.ifmask:
                return self.mask(input,s,fs),self.alpha
            else:
                return fs,self.alpha

        if (self.alpha < 0.0):
            fs = - torch.log(1 - self.alpha * (s + self.beta)) / self.alpha
            if self.ifmask:
                return self.mask(input,s,fs),self.alpha
            else:
                return fs,self.alpha

        if (self.alpha > 0.0):
            fs = (torch.exp(self.alpha * s) - 1)/ self.alpha + self.alpha
            if self.ifmask:
                return self.mask(input,s,fs),self.alpha
            else:
                return fs,self.alpha