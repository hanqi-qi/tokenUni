#rescaling function for the eigenvalue shift/scale
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F
class soft_decay_function(nn.Module):
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
    def __init__(self, in_features, alpha = None, hidden_dim=768, decay_alpha=-0.2,ifmask=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_decay_function,self).__init__()
        self.in_features = in_features
        self.ifmask = ifmask

        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(decay_alpha)) # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha
        self.alpha.requiresGrad = True # set requiresGrad to true!

        self.activations = {'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'relu': torch.relu, 'leaky_relu': F.leaky_relu}
        self.activation = self.activations["relu"]

    

    def forward(self, input,s):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        eps = 1e-7
        fs = - torch.log(1 - self.alpha * (s + self.alpha)+eps) / self.alpha
        return fs,self.alpha