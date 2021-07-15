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

def tokenSimi(tokens_matrix,seqlen=None,nopad=False):
    """calculate the average cosine similarity,with/without normalization"""
    simi = []
    cls_simi = []
    if nopad:
        l = seqlen
    else:
        l = tokens_matrix.shape[0]
    for i in range(l):
        for j in range(l):
            if i!=j:
                simi.append(dot(tokens_matrix[i],tokens_matrix[j])/(norm(tokens_matrix[i])*norm(tokens_matrix[j])))
    for i in range(l):
        cls_simi.append(dot(tokens_matrix[0],tokens_matrix[i])/(norm(tokens_matrix[0])*norm(tokens_matrix[i])))
    return sum(simi)/len(simi),sum(cls_simi)/len(cls_simi)
#TODO(yhq:0629): add metrics evaluating the eigenvalue distribution
def rbf_kernel(input,unif_t):
    sq_pdist = torch.pdist(input, p=2).pow(2)
    uni_loss = sq_pdist.mul(-unif_t).exp().mean().log() #results are negative one, the maximal value is zero, for visualization, using the exp(uni_loss) to transform the range to [0,1]
    return uni_loss

def Evs(input,k):
    """calcualte EV_{k}(h) from https://arxiv.org/pdf/2005.02178.pdf"""
    _,s,_ = torch.svd(input)
    ek = [s[i]*s[i] for i in range(k)]
    ed = [s[i]*s[i] for i in range(len(s)-k,len(s))]
    return sum(ek)/sum(ed)
def multiCDF():
    dataset_name = "mrpc"
    model_type = "bert-base-uncased"
    pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/".format("glue",model_type)
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    plt.subplots(figsize=(8, 4))
    colors = ["blue","red","cyan","black","green"]
    # #draw baseline cdf
    # basicDataPath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/mrpc/bert-base-uncased/weight/origin/fixedweight_Epoch0_None.npy"
    # basicData = np.load(basicDataPath)
    # sample_i = 0
    # layer_i = -1
    # sv, _ = singular_spectrum(basicData[sample_i,layer_i,:,:])
    # arr_1_edf = np.arange(1, 1+len(sv), 1)/len(sv)
    # arr_1_sorted = np.sort(sv/max(sv))#normalize to 1
    # plt.plot(arr_1_sorted, arr_1_edf, label='W.o fine-tuning',color = "black",linewidth=1)
    for lnv in ["origin","soft_expand"]:
        for epoch in ["0","1","2"]:
        #read the npy data and draw the CDF curve
            if lnv == "origin":
                datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,lnv,epoch)
            elif lnv == "soft_expand":
                datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_add_last_beforeln.npy".format(dataset_name,model_type,lnv,epoch)
            basicData = np.load(datapath)
            sample_i = 0
            for layer_i in basicData.shape[0]:
                if layer_i%3==0:
                    sv, _ = singular_spectrum(basicData[sample_i,layer_i,:,:])
                    arr_1_edf = np.arange(1, 1+len(sv), 1)/len(sv)
                    arr_1_sorted = np.sort(sv/max(sv))
                    plt.plot(arr_1_sorted, arr_1_edf, label='%sLayer%d'%(lnv,layer_i),color = colors[int(layer_i/3)],linewidth=1)
    plt.legend()
    plt.title("CDF for %s Task at Different Layers" %(dataset_name))
    plt.ylabel("F(x)")
    plt.xlabel('x')
    child_dir = "LastEpoch_{}AllLayer_cdf_average.pdf".format(epoch)
    pic_name = os.path.join(pic_dir,child_dir)
    plt.savefig(pic_name,dpi=500)

#TODO(yhq0528): 
def draw_cdf(hidden_outputs,args,n_bins,metrics):
    sample_i = 0
    pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}/".format(args.dataset_name,args. model_type,args.lnv1)
    #TODO(yhq0528):calculate the average of eigenvalue distribution
    average_batchs = True
    if args.mode == "compare":
        sample_i = 0
        W1,W2 = hidden_outputs[0][sample_i,:,:,:],hidden_outputs[1][sample_i,:,:,:]
        for i_layer in range(W1.shape[0]):
            arr_1, _ = singular_spectrum(W1[i_layer,:,:])
            arr_2, _ = singular_spectrum(W2[i_layer,:,:])
            arr_1_edf = np.arange(1, 1+len(arr_1), 1)/len(arr_1)
            arr_1_sorted = np.sort(arr_1)#sort
            
            plt.plot(arr_1_sorted, arr_1_edf, label='Baseline: AUC%.2f;MiniEigen%.2f'%(auc1,arr_1_sorted[0]),color = "black",linestyle='dashed',linewidth=1)
            arr_2_edf = np.arange(1, 1+len(arr_2), 1)/len(arr_2)
            arr_2_sorted = np.sort(arr_2)#sort
            plt.plot(arr_2_sorted, arr_2_edf, label='with_posEmb: AUC%.2f; MiniEigen%.2f'%(auc2,arr_2_sorted[0]),color="black",linestyle='solid',linewidth=1)
            arr_dif_abs = np.abs(arr_2_edf-arr_1_edf)

            plt.legend()
            plt.title("Cumulative Distribution Functions for %s Task"%dataset_name)
            plt.ylabel("F(x)")
            plt.xlabel('x')

            pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}/".format(args.dataset_name,args. model_type,args.lnv1)
            child_path = "{}v.s.{}_Epoch_{}Layer_{}Apply_{}Nonlinear_{}_cdf2116.png".format(args.lnv1,args.lnv2,args.epoch,i_layer,args.apply_exrank2,"relu")
            pic_name = os.path.join(pic_dir,child_path)
            plt.savefig(pic_name)
            plt.close('all')


def draw_hist(hidden_outputs,args,n_bins,metrics,ifcdf=None):
    #all the layers, sample_i = 0,
    sample_i = 0
    pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}/".format(args.dataset_name,args. model_type,args.lnv1)
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
                        tokenUni1,clsUni1 = tokenSimi(hidden_outputs[0][sample_i,i_layer,:,:])
                        tokenuni.append(tokenUni1)
                        clsuni.append(clsUni1)
            else:
                W = hidden_outputs[0][sample_i,:,:,:] #[i_layer, seq_len, f]
                tokenUni1,clsUni1 = tokenSimi(W[i_layer,:,:])
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
            child_dir = "a0Epoch_{}Layer_{}Apply_{}Nonlinear_pdf_average_annotate.png".format(args.epoch,i_layer,args.apply_exrank1)
            pic_name = os.path.join(pic_dir,child_dir)
            fig, axs = plt.subplots(figsize=(8, 5))
            w1_stats= stats.describe(sv)
            plt.title("PDF for {} at Layer {}".format(args.dataset_name,i_layer))
            sns.distplot(sv/max(sv), hist=True, kde=True, 
                bins=50, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2,"bw_adjust":0.1})
            # plt.hist(sv4, 50, facecolor='green', alpha=0.5)
            # ax.hist(np.abs(sv/max(sv)), n_bins)
            
            axs.text(axs.get_xlim()[1]*0.65,axs.get_ylim()[1]*0.6,"variance:"+str("%.2f"%w1_stats.variance)+"\n"+"skewness:"+str("%.2f"%w1_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w1_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%(sum(tokenuni)/len(tokenuni)))+"\n"+"clsUni:"+str("%.2f"%(sum(clsuni)/len(clsuni))),fontsize = 14)
            # axs.set_yscale('log')
            plt.xlabel("singular value",size=14)
            plt.ylabel("#singular value",size=14)
            axs.tick_params(axis='x', labelsize=14)
            axs.tick_params(axis='y', labelsize=14)
            plt.savefig(pic_name)
        #     if ifcdf==True: #draw cdf
        #         arr_1_edf = np.arange(1, 1+len(sv), 1)/len(sv)
        #         arr_1_sorted = np.sort(sv/max(sv))#normalize to 1
        #         if i_layer%3 == 0:
        #             plt.plot(arr_1_sorted, arr_1_edf, label='Layer%d'%(i_layer),color = colors[int(i_layer/3)],linewidth=1)
        # plt.legend()
        # plt.title("CDF for %s Task at Different Layers" %(args.dataset_name))
        # plt.ylabel("F(x)")
        # plt.xlabel('x')
        # child_dir = "Epoch_{}AllLayer_Apply_{}Nonlinear_cdf_average.pdf".format(args.epoch,args.apply_exrank1)
        # pic_name = os.path.join(pic_dir,child_dir)
        # plt.savefig(pic_name,dpi=500)

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
            #calculate token_uniformity for sentence-level 
            tokenUni1,clsUni1 = tokenSimi(W1[i_layer,:,:])
            tokenUni2,clsUni2 = tokenSimi(W2[i_layer,:,:])
            w1_stats= stats.describe(sv1)
            w2_stats= stats.describe(sv2)
            child_path = "0602{}v.s.{}_Epoch_{}Layer_{}Apply_{}Nonlinear_{}_hist_normalizeTo1.png".format(args.lnv1,args.lnv2,args.epoch,i_layer,args.apply_exrank2,"relu")
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
            # axs[0].hist(np.abs(sv1/max(sv1)), n_bins)
            # axs[1].hist(np.abs(sv2/max(sv2)), n_bins)
            #draw the pdf
            sns.distplot(sv1/max(sv1), hist=True, kde=True, 
                bins=50, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2,"bw_adjust":0.1},ax=axs[0])
            sns.distplot(sv2/max(sv2), hist=True, kde=True, 
                bins=50, color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 2,"bw_adjust":0.1},ax=axs[1])
            #mark the median line
            median1 = np.percentile(sv1/max(sv1), 50)
            median2 = np.percentile(sv2/max(sv2), 50)
            axs[0].vlines(x=median1,ymin=0,ymax=0.98*axs[0].get_ylim()[1],color="red",label="median=%.4f"%median1)
            axs[0].legend()
            axs[1].vlines(x=median2,ymin=0,ymax=0.98*axs[1].get_ylim()[1],color="red",label="median=%.4f"%median2)
            axs[1].legend()
            # axs[0].grid()
            # axs[1].grid()
            # axs[0].set_yscale('log')
            # axs[1].set_yscale('log')
            axs[1].tick_params(axis='x', labelsize=14)
            axs[1].tick_params(axis='y', labelsize=14)
            axs[0].tick_params(axis='y', labelsize=14)
            pic_name = os.path.join(pic_dir,child_path)
            
            
            # ax[0].set_xticklabels(xlabels, Fontsize=28 )
            # axs[0].set_title("%s rank:%d"%(args.lnv1,rank1))
            # axs[0].text(axs[0].get_xlim()[1]*0.65,axs[0].get_ylim()[1]/7,"mean:"+str("%.2f"%w1_stats.mean)+"\n"+"maxmin:("+str("%.2f"%w1_stats.minmax[0])+","+str("%.2f"%w1_stats.minmax[1])+")"+"\n"+"variance:"+str("%.2f"%w1_stats.variance)+"\n"+"skewness:"+str("%.2f"%w1_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w1_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%tokenUni1)+"\n"+"clsUni:"+str("%.2f"%clsUni1))
            axs[0].text(axs[0].get_xlim()[1]*0.60,axs[0].get_ylim()[1]/15,"variance:"+str("%.2f"%w1_stats.variance)+"\n"+"skewness:"+str("%.2f"%w1_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w1_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%tokenUni1)+"\n"+"clsUni:"+str("%.2f"%clsUni1),fontsize = 15)
            axs[1].text(axs[1].get_xlim()[1]*0.60,axs[1].get_ylim()[1]/12,"variance:"+str("%.2f"%w2_stats.variance)+"\n"+"skewness:"+str("%.2f"%w2_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w2_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%tokenUni2)+"\n"+"clsUni:"+str("%.2f"%clsUni2),fontsize = 15)
            plt.savefig(pic_name)
            # axs[1].set_title("%s rank:%d"%(args.lnv2,rank2))
            # print("*****Layer%d"%i_layer)
            # print("mean:"+str("%.2f"%w1_stats.mean)+"\n"+"maxmin:("+str("%.2f"%w1_stats.minmax[0])+","+str("%.2f"%w1_stats.minmax[1])+")"+"\n"+"variance:"+str("%.2f"%w1_stats.variance)+"\n"+"skewness:"+str("%.2f"%w1_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w1_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%tokenUni1)+"\n"+"clsUni:"+str("%.2f"%clsUni1))
            # print("!!!!")
            # axs[1].set_title("%s rank:%d"%(args.lnv2,rank2))
            # print("mean:"+str("%.2f"%w2_stats.mean)+"\n"+"maxmin:("+str("%.2f"%w2_stats.minmax[0])+","+str("%.2f"%w2_stats.minmax[1])+")"+"\n"+"variance:"+str("%.2f"%w2_stats.variance)+"\n"+"skewness:"+str("%.2f"%w2_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w2_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%tokenUni2)+"\n"+"clsUni:"+str("%.2f"%clsUni2))
            # axs[1].text(axs[1].get_xlim()[1]*0.65,axs[1].get_ylim()[1]/7,"mean:"+str("%.2f"%w2_stats.mean)+"\n"+"maxmin:("+str("%.2f"%w2_stats.minmax[0])+","+str("%.2f"%w2_stats.minmax[1])+")"+"\n"+"variance:"+str("%.2f"%w2_stats.variance)+"\n"+"skewness:"+str("%.2f"%w2_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%w2_stats.kurtosis)+"\n"+"tokenuni:"+str("%.2f"%tokenUni2)+"\n"+"clsUni:"+str("%.2f"%clsUni2))
    #         if ifcdf==True: #draw cdf
    #             arr_1_edf = np.arange(1, 1+len(sv1), 1)/len(sv1)
    #             arr_1_sorted = np.sort(sv1/max(sv1))#normalize to 1
    #             arr_2_edf = np.arange(1, 1+len(sv2), 1)/len(sv2)
    #             arr_2_sorted = np.sort(sv1/max(sv2))#normalize to 1
    #             if i_layer%3 == 0:
    #                 plt.plot(arr_1_sorted, arr_1_edf, label='Layer%d'%(i_layer),color = colors[int(i_layer/3)],linewidth=1)
    #                 plt.plot(arr_2_sorted, arr_2_edf, label='Layer%d'%(i_layer),color = colors[-int(i_layer/3)],linewidth=1)

    #     plt.legend()
    #     plt.title("CDF for %s Task at Different Layers" %(args.dataset_name))
    #     plt.ylabel("F(x)")
    #     plt.xlabel('x')
    #     child_dir = "Epoch_{}AllLayer_Apply_{}Nonlinear_cdf_average.pdf".format(args.epoch,args.apply_exrank1)
    #     pic_name = os.path.join(pic_dir,child_dir)
    #     plt.savefig(pic_name,dpi=500)
    #     plt.savefig(pic_name)
    #     plt.close('all')
    # else:
    #     print("please keep the mode and input matrix compatiable!")


# def draw_cdf(config=None,arr=None,epoch=None,picName = None,dataset = None):
#     #parse the information from the input npy filename.
#     plt.figure(figsize = (10, 5))
#     if not isinstance(arr, list):
#         arr_1 = arr
#         arr_1_edf = np.arange(1, 1+len(arr_1), 1) / len(arr_1)
#         arr_1_sorted = np.sort(arr_1)
#         plt.plot(arr_1_sorted, arr_1_edf, label='F_obs')
#     else: #TODO(yhq): need to specif the two comparisons
#         arr_1 = arr[0]
#         arr_2 = arr[1]
#         arr_1_edf = np.arange(1, 1+len(arr_1), 1)/len(arr_1)
#         arr_1_sorted = np.sort(arr_1)#sort
#         plt.plot(arr_1_sorted, arr_1_edf, label='Baseline: AUC%.2f;MiniEigen%.2f'%(auc1,arr_1_sorted[0]),color = "black",linestyle='dashed',linewidth=1)
#         arr_2_edf = np.arange(1, 1+len(arr_2), 1)/len(arr_2)
#         arr_2_sorted = np.sort(arr_2)#sort
#         plt.plot(arr_2_sorted, arr_2_edf, label='with_posEmb: AUC%.2f; MiniEigen%.2f'%(auc2,arr_2_sorted[0]),color="black",linestyle='solid',linewidth=1)
#         arr_dif_abs = np.abs(arr_2_edf-arr_1_edf)

#     plt.legend()
#     plt.title("Cumulative Distribution Functions for %s Task"%args.dataset_name)
#     plt.ylabel("F(x)")
#     plt.xlabel('x')
#     if not isinstance(arr, list):
#         parent_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}".format(config.dataset_name,config.model_name_or_path,config.lnv)
#         pic_path =  "Epoch_{}Apply_{}Nonlinear_{}_cdf.png".format(epoch,config.apply_exrank,config.exrank_nonlinear)
#         pic_name = os.path.join(parent_dir,pic_path)
#         if not os.path.isdir(parent_dir):
#             os.makedirs(parent_dir)
#         plt.title("Epoch {} sample {} layer {}".format(epoch,sample_i,i_layer))
#     else:
#         parent_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}".format(config.dataset_name,config.model_name_or_path,config.lnv)
#         pic_path =  "Epoch_{}Apply_{}Nonlinear_{}_cdf.png".format(epoch,config.apply_exrank,config.exrank_nonlinear)
#         pic_name = os.path.join(parent_dir,pic_path)
#         if not os.path.isdir(parent_dir):
#             os.makedirs(parent_dir)
#     plt.savefig(pic_name)

#model,dataset,strategy/lnv,output for layers,rank,moments,performance
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
    # ho1 = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch_{}Apply_{}Nonlinear_relu_new.npy".format(args.dataset_name,args.model_type,args.lnv1,args.epoch, args.apply_exrank1)
    dataset_name = "rte"
    model_type = "bert-base-uncased"
    basicDataPath = datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,"origin","2")
    hidden_outputs.append(np.load(basicDataPath))
    if not args.lnv2 == None:    
        ho2 = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch_{}Apply_{}Nonlinear_relu_new.npy".format(args.dataset_name,args.model_type,args.lnv2,args.epoch, args.apply_exrank2)
        hidden_outputs.append(np.load(ho2))

    # draw_hist(hidden_outputs, args=args,n_bins=50,metrics=None)
    draw_hist(hidden_outputs, args=args,n_bins=50,metrics=None,ifcdf=False)

