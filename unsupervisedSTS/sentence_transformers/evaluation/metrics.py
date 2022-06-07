from numpy import dot, square
from numpy.linalg import norm
import torch
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold
from sklearn.manifold import locally_linear_embedding
from scipy.linalg import eigh, svd, qr, solve
import numpy as np

def barycenter_weights(X, Y, indices, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis

    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i]. The barycenter weights sum to 1.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)

    Y : array-like, shape (n_samples, n_dim)

    indices : array-like, shape (n_samples, n_dim)
            Indices of the points in Y used to compute the barycenter

    reg : float, default=1e-3
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim

    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)

    Notes
    -----
    See developers note for more information.
    """
    # X = check_array(X, dtype=FLOAT_DTYPES)
    # Y = check_array(Y, dtype=FLOAT_DTYPES)
    # indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace
    # zero
    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::n_neighbors + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B

def tokenSimi(input):
    """calculate the average cosine similarity,with/without normalization"""
    norm_input = input/torch.norm(input,dim=-1,keepdim=True)
    simi_matrix = torch.matmul(norm_input,norm_input.t())
    return torch.sum(simi_matrix)/(input.shape[0]*input.shape[0])
#TODO(yhq:0629): add metrics evaluating the eigenvalue distribution
def rbf_kernel(input,unif_t):
    norm_input = input/torch.norm(input,dim=-1,keepdim=True)
    sq_pdist = torch.pdist(norm_input, p=2).pow(2)
    uni_loss = sq_pdist.mul(-unif_t).exp().mean().log() #results are negative one, the maximal value is zero, for visualization, using the exp(uni_loss) to transform the range to [0,1]
    return uni_loss

def Evs(input,k):
    """calcualte EV_{k}(h) from https://arxiv.org/pdf/2005.02178.pdf"""
    try:
        _,s,_ = torch.svd(input)
        ek = [s[i]*s[i] for i in range(k)]
        square_sum = sum(s*s)
        result = sum(ek)/square_sum
    except:
        print("svd fails!")
        result = 0.0
    return result

def struct_loss(input,k):
    """calculate the k-nearest points for each point"""
    norm_input = input/torch.norm(input,dim=-1,keepdim=True)
    simi_matrix = torch.matmul(norm_input,norm_input.t())
    _,index=torch.topk(simi_matrix, k)
    mask = torch.zeros_like(simi_matrix)
    for row in range(input.shape[0]):
        mask[row][index[row]]=1
    return mask


def test(X):
    X_data = X.detach().cpu().numpy()
    X_re, error = locally_linear_embedding(X_data,n_components=768,n_neighbors=12)
    return error

def nearbyloss(X,Xt):
    X_data = X.detach().cpu().numpy()
    X_data = X_data/norm(X_data, ord=2, axis=1, keepdims=True)
    Xt_data = Xt.detach().cpu().numpy()
    Xt_data = Xt_data/norm(Xt_data, ord=2, axis=1, keepdims=True)
    n_components=  768
    n_neighbors = 12

    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs_.fit(X_data)
    dis,ind = nbrs_.kneighbors(X_data, n_neighbors=n_neighbors,
                                    return_distance=True)
    """use the nearset points from indexed by ind to reconstruct the X_data, return the weight 
    1)nbrs_._fit_X == X_data. 
    2)if exclude_self, the Knearest set does NOT include the query point."""
    # exclude_self = True
    # if exclude_self:
    #     ind = ind[:,1:]
    # else:
    #     ind = ind
    weights = barycenter_weights(X_data, nbrs_._fit_X, ind)
    #reconstruct the origin_data and compute the origin_error
    re_X = np.empty((X_data.shape[0], n_components))
    for i in range(X_data.shape[0]):
        re_X[i] = np.dot(nbrs_._fit_X[ind[i]].T, weights[i])
    error = norm(X_data- re_X, 'fro')
    #represents the new data using the same i-th data and compute the origin_error
    re_Xt = np.empty((X_data.shape[0], n_components))
    for i in range(X_data.shape[0]):
        re_Xt[i] = np.dot(Xt_data[ind[i]].T, weights[i])
    t_error = norm(re_Xt-Xt_data, 'fro')
    return error,t_error
