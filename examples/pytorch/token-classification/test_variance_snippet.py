"""Input:hidden_states, outoput the moment of the tensor"""
from scipy import stats
batch_id = 0
n_layer = -1
u1,s1,v1 = torch.svd(hidden_states[batch_id,n_layer])
s1 = s1.cpu().detach().numpy()
norm_s1 = s1/s1.max()
w1_stats= stats.describe(norm_s1)
