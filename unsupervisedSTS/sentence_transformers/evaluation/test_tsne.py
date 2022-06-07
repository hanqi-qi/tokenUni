from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# embeddings1 = np.random.rand(10,500)
# embeddings2 = np.random.rand(10,500)*
# input = np.concatenate((embeddings1,embeddings2),axis=0)
# ifpca = False
# if ifpca:
#     pca_50 = PCA(n_components=50)
#     pca_result_50 = pca_50.fit_transform(input)
# else:
#     pca_result_50 = input
# tokens = pca_result_50
tokens = np.random.randn(20,2)
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=330)
# tsne_results = tsne.fit_transform(tokens)
categories = 10*["0"]+10*["1"]
d = {'tsne-2d-one': tokens[:,0], 'tsne-2d-two':tokens[:,1],"color_cate":categories}
df_subset = pd.DataFrame(data=d)
fig,ax = plt.subplots(figsize=(6, 6))
colors = {'0':'red', '1':'black'}
ax.scatter(df_subset['tsne-2d-one'], df_subset['tsne-2d-two'], c=df_subset['color_cate'].map(colors))

# plt.show()
# df_subset.plot.scatter(x="tsne-2d-one",y="tsne-2d-two",s=25,ax=ax,c="color_cate",colormap="Greens")
# df_subset.plot.scatter(x="tsne-2d-one",y="tsne-2d-two",s=25,ax=ax,c="red")
plt.title("tsne for sentences1 and sentences2")
picname = "tesne.png"
plt.savefig(picname)