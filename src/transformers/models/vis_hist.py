from numpy.core.fromnumeric import size
from numpy.lib.function_base import average
from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import math
from astropy.visualization import hist #flexible bins setting
from scipy.stats import skew
import os
import argparse
from scipy import stats
from fitter import Fitter
from sklearn import preprocessing
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import torch.nn.functional as F

"""Based on the vis_tools, but visualize the eigenvalues/rank derived from the saved_matrix in .npy file. Functions should include as main data(legend):
1) single output: hist/cdf chart (model,dataset,strategy,output for layers,rank,moments,performance);
2) multiple charts:
2.1 different train epochs 
2.2 different strategies along the same axis
"""


# root_dir = "/mnt/Data3/hanqiyan/rank_transformers"
def singular_spectrum(W, norm=False): 
    if norm:
        W = W/np.trace(W)
    M = np.min(W.shape)
    svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
    svd.fit(W) 
    svals = svd.singular_values_
    svecs = svd.components_
    return svals, svecs

def draw_hist(hidden_outputs,args,n_bins,metrics,ifcdf=None):
    #all the layers, sample_i = 0,
    sample_i = 0
    pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}/".format(args.dataset_name,args.model_type,args.lnv1)
    #TODO(yhq0528):calculate the average of eigenvalue distribution
    colors = ["blue","red","cyan","black","green"]
    # markers = [".","D","s","x"]
    # line_types = []
    average_batchs = True
    plt.subplots(figsize=(8, 4))
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    if args.mode == "single" and len(hidden_outputs) == 1:
        for i_layer in range(hidden_outputs[0][0,:,:,:].shape[0]):
            if average_batchs:
                sv = []
                tokenuni,clsuni = [],[]
                for sample_i in range(hidden_outputs[0].shape[0]):
                        sv_batch, _ = singular_spectrum(hidden_outputs[0][sample_i,i_layer,:,:])
                        sv.extend(sv_batch)
            else:
                W = hidden_outputs[0][sample_i,:,:,:] #[i_layer, seq_len, f]
                # tokenUni1,clsUni1 = tokenSimi(W[i_layer,:,:])
                for i_layer in range(W.shape[0]):
                    W_i = W[i_layer,:,:]
                    sv, _ = singular_spectrum(W_i)
            #keep only one figure handler at once is better
            print("*****")
            print(i_layer)
            print(np.percentile(sv/max(sv), 10))
            print(np.percentile(sv/max(sv), 25))
            print(np.percentile(sv/max(sv), 50))
            print("*****")
            child_dir = "{}_Layer{}_.png".format(args.model_type,i_layer,args.lnv1)
            pic_name = os.path.join(pic_dir,child_dir)
            fig, axs = plt.subplots(figsize=(8, 5))
            w1_stats= stats.describe(sv)
            plt.title("PDF for {} at Layer {}".format(args.dataset_name,i_layer))
            sns.distplot(sv/max(sv), hist=True, kde=True, 
                bins=50, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2,"bw_adjust":0.1})         
            axs.text(axs.get_xlim()[1]*0.65,axs.get_ylim()[1]*0.6,"variance:"+str("%.2f"%w1_stats.variance)+"\n"+"skewness:"+str("%.2f"%w1_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w1_stats.kurtosis)+"\n",fontsize = 14)
            # axs.set_yscale('log')
            plt.xlabel("singular value",size=14)
            plt.ylabel("#singular value",size=14)
            axs.tick_params(axis='x', labelsize=14)
            axs.tick_params(axis='y', labelsize=14)
            plt.savefig(pic_name)
    elif args.mode == "compare" and len(hidden_outputs) > 1:
        if len(hidden_outputs[0].shape) == 3:
            W1=hidden_outputs[0].reshape(-1,13,hidden_outputs[0].shape[-2],hidden_outputs[0].shape[-1])[sample_i,:,:,:]
            W2=hidden_outputs[1].reshape(-1,13,hidden_outputs[1].shape[-2],hidden_outputs[1].shape[-1])[sample_i,:,:,:]
        elif len(hidden_outputs[0].shape) == 4:
            W1,W2 = hidden_outputs[0][sample_i,:,:,:],hidden_outputs[1][sample_i,:,:,:]
        for i_layer in range(W1.shape[0]):
            sv1, _ = singular_spectrum(W1[i_layer,:,:])
            rank1 = np.linalg.matrix_rank(W1[i_layer,:,:])
            sv2, _ = singular_spectrum(W2[i_layer,:,:])
            rank2 = np.linalg.matrix_rank(W2[i_layer,:,:])
            w1_stats= stats.describe(sv1)
            w2_stats= stats.describe(sv2)
            child_path = "0713Layer{}_{}vs{}_hist_normalizeTo1.png".format(i_layer,args.lnv1,args.lnv2)
            pic_name = os.path.join(pic_dir,child_path)
            plt.figure(figsize = (16, 12))
            fig, axs = plt.subplots(2, 1, sharex=True)#common x-axis
            #add grid between subfigs
            ax3 = fig.add_subplot(111, zorder=-1)
            for _, spine in ax3.spines.items():
                spine.set_visible(False)
            ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False )
            ax3.get_shared_x_axes().join(ax3,axs[0])
            ax3.grid(axis="x")
            #common title
            fig.suptitle("%s %s Layer:%d"%(args.dataset_name,args.model_type,i_layer))
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0.05)
            sns.distplot(sv1, hist=True, kde=True, 
                bins=50, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2,"bw_adjust":0.1},ax=axs[0],label="BERT")
            sns.distplot(sv2, hist=True, kde=True, 
                bins=50, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2,"bw_adjust":0.1},ax=axs[1],label="BERT+SoftDecay")
            # #mark the median line
            # median1 = np.percentile(sv1/max(sv1), 50)
            # median2 = np.percentile(sv2/max(sv2), 50)
            # axs[0].vlines(x=median1,ymin=0,ymax=0.98*axs[0].get_ylim()[1],color="red",label="median=%.4f"%median1)
            axs[0].legend()
            # axs[1].vlines(x=median2,ymin=0,ymax=0.98*axs[1].get_ylim()[1],color="red",label="median=%.4f"%median2)
            axs[1].legend()
            axs[1].tick_params(axis='x', labelsize=14)
            axs[1].tick_params(axis='y', labelsize=14)
            axs[0].tick_params(axis='y', labelsize=14)
            pic_name = os.path.join(pic_dir,child_path)
            
    
            axs[0].text(axs[0].get_xlim()[1]*0.60,axs[0].get_ylim()[1]/3,"minValue:"+str("%.2f"%min(sv1))+"\n"+"maxValue:"+str("%.2f"%max(sv1))+"\n""variance:"+str("%.2f"%w1_stats.variance),fontsize = 15)
            axs[1].text(axs[1].get_xlim()[1]*0.60,axs[1].get_ylim()[1]/3,"minValue:"+str("%.2f"%min(sv2))+"\n"+"maxValue:"+str("%.2f"%max(sv2))+"\n""variance:"+str("%.2f"%w2_stats.variance),fontsize = 15)
            plt.savefig(pic_name)


def parse_args():
    parser = argparse.ArgumentParser(description="accept the input arguments to specify the display matrix")

    parser.add_argument("--model_type", type=str,default="bert-base-uncased",
        help="model_type"
    )
    parser.add_argument("--dataset_name", type=str,default="conll2003",
        help="conll2003/mrpc/squad/swag/wikitext"
    )
    parser.add_argument("--mode", type=str,default = "single",help="single model or comparing different"
    )
    parser.add_argument("--lnv1", type=str,default = "origin", help="different normalization strategies or origin/layernorm"
    )
    parser.add_argument("--lnv2", type=str,default = None, help="different normalization strategies or origin/layernorm"
    )

    parser.add_argument("--apply_exrank1", type=str,default = None, help="different implement for exrank_gx"
    )
    parser.add_argument("--apply_exrank2", type=str,default = None, help="different implement for exrank_gx"
    )
    parser.add_argument("--epoch", type=int,default = 0, help="vis hidden_states in different epochs "
    )

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    #read the savad_matrix based on the input arguments
    hidden_outputs = []
    dataset_name = args.dataset_name
    model_type = args.model_type
    basicDataPath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,args.lnv1,"2")
    hidden_outputs.append(np.load(basicDataPath))
    if not args.lnv2 == None:    
        ho2  = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_add_last_beforeln.npy".format(dataset_name,model_type,args.lnv2,"2")
        hidden_outputs.append(np.load(ho2))

    # draw_hist(hidden_outputs, args=args,n_bins=50,metrics=None)
    draw_hist(hidden_outputs, args=args,n_bins=50,metrics=None,ifcdf=False)