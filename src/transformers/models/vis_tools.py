from sklearn.decomposition import TruncatedSVD
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import math
import scipy.stats as st
# from astropy.visualization import hist #flexible bins setting
from scipy.stats import skew
import os

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

def evaluate_dis(sv):
    metrics = {"skewness":0,"peakness":0}
    return metrics

def draw_cdf(config=None,arr=None,sample_i=None,i_layer=None,epoch=None,picName = None,auc1=0,auc2=0,dataset = None):
    #edf
    plt.figure(figsize = (10, 5))
    if not isinstance(arr, list):
        arr_1 = arr
        arr_1_edf = np.arange(1, 1+len(arr_1), 1) / len(arr_1)
        arr_1_sorted = np.sort(arr_1)
        plt.plot(arr_1_sorted, arr_1_edf, label='F_obs')
    else: #TODO(yhq): need to specif the two comparisons
        arr_1 = arr[0]
        arr_2 = arr[1]
        arr_1_edf = np.arange(1, 1+len(arr_1), 1)/len(arr_1)
        arr_1_sorted = np.sort(arr_1)#sort
        plt.plot(arr_1_sorted, arr_1_edf, label='Baseline: AUC%.2f;MiniEigen%.2f'%(auc1,arr_1_sorted[0]),color = "black",linestyle='dashed',linewidth=1)
        arr_2_edf = np.arange(1, 1+len(arr_2), 1)/len(arr_2)
        arr_2_sorted = np.sort(arr_2)#sort
        plt.plot(arr_2_sorted, arr_2_edf, label='with_posEmb: AUC%.2f; MiniEigen%.2f'%(auc2,arr_2_sorted[0]),color="black",linestyle='solid',linewidth=1)
        arr_dif_abs = np.abs(arr_2_edf-arr_1_edf)

    plt.legend()
    plt.title("Cumulative Distribution Functions for %s Task"%dataset)
    plt.ylabel("F(x)")
    plt.xlabel('x')
    if not isinstance(arr, list):
        parent_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}".format(config.dataset_name,config.model_name_or_path,config.lnv)
        pic_path =  "Epoch_{}Apply_{}Nonlinear_{}_cdf.png".format(epoch,config.apply_exrank,config.exrank_nonlinear)
        pic_name = os.path.join(parent_dir,pic_path)
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
        plt.title("Epoch {} sample {} layer {}".format(epoch,sample_i,i_layer))
    else:
        parent_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}".format(config.dataset_name,config.model_name_or_path,config.lnv)
        pic_path =  "Epoch_{}Apply_{}Nonlinear_{}_cdf.png".format(epoch,config.apply_exrank,config.exrank_nonlinear)
        pic_name = os.path.join(parent_dir,pic_path)
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
    plt.savefig(pic_name)

def save_matrix(input_tensor,epoch,config,mode="train",timestamp='new'):
    #input_tensor: [n_samples,layers,sequen_len,dim]
    W = np.array(input_tensor.data.clone().cpu()) #
    if mode == "train":
        epoch = str(epoch)
    elif mode == 'eval':
        epoch = "EVAL:"+str(epoch)
    parent_dir = "//home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/weight/{}".format(config.dataset_name,config.model_name_or_path,config.lnv)
    matrix_path =  "Epoch{}_{}.npy".format(epoch,config.apply_exrank)
    matrix_name = os.path.join(parent_dir,matrix_path)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    np.save(matrix_name,W) #np.load(OutputTensor)

def PlotEigen(input_tensor,epoch,config,mode="train",predict_metric=None):
    W = np.array(input_tensor.data.clone().cpu()) #
    W = np.concatenate(W,axis=0)
    min_svs = []
    for sample_i in range(W.shape[0]): #number of sequences
        if sample_i%1000==0:
            W_sample = W[sample_i]
            for i_layer in range(W_sample.shape[0]):
                W_i = W_sample[i_layer,:,:]
                M, N = np.min(W_i.shape), np.max(W_i.shape)#recutangle
                Q=N/M #aspect ratio
                sv, _ = singular_spectrum(W_i)
                rank = np.linalg.matrix_rank(W_i)
                print("rank is %d"%rank)
                parent_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/{}".format(config.dataset_name,config.model_name_or_path,config.lnv)
                pic_path =  "Epoch_{}Apply_{}Nonlinear_{}_hist.png".format(epoch,config.apply_exrank,config.exrank_nonlinear)
                pic_name = os.path.join(parent_dir,pic_path)
                draw_cdf(config,sv,sample_i,i_layer,epoch,picName=None)#draw cdf for distribution
                if not os.path.isdir(parent_dir):
                    os.makedirs(parent_dir)
                n_bins = 50
                fig, ax = plt.subplots(figsize=(8, 4))
                plt.title("Epoch {} sample {} layer {} Rank {}".format(epoch,sample_i,i_layer,rank))
                n, bins, patches = ax.hist(np.abs(sv), n_bins)
                if mode == "train":
                    plt.legend("train")
                else:
                    plt.legend("%s:%2f"%(list(predict_metric.keys())[0],predict_metric[list(predict_metric.keys())[0]]))
                ax.set_yscale('log')
                plt.savefig(pic_name)

if __name__=="__main__":
    sample_i = 0
    i_layer = 5
    seqlen = 80
    dim = 84
    depth = 6
    auc1 = 0.0 
    auc2 = 0.0
    task1 = "baseline"
    task2 = "wpos"
    use_squareM =  False
    if_sc = True
    dataset = "convexhull"
    # m1 = np.load("Epoch_1.0_output.npy")[sample_i,i_layer]
    # m2 = np.load("Epoch_16.0_output.npy")[sample_i,i_layer]
    m1_file = "hull_skiptrain/{}/weight/EpochEVALDepth{}Len{}Hidden{}_{}_output.npy".format(
            task1,depth,seqlen,dim,"wo_sc" if if_sc else "with_sc")
    m2_file = "hull_skiptrain/{}/weight/EpochEVALDepth{}Len{}Hidden{}_{}_output.npy".format(
            task2,depth,seqlen,dim,"wo_sc" if if_sc else "with_sc")
    m1 = np.load(m1_file)[sample_i,i_layer]
    m2 = np.load(m2_file)[sample_i,i_layer]
    # m2 = np.load("hull_skiptrain/wpos/weight/EpochEVALDepth2Len80Hidden84_wo_sc_output.npy")[sample_i,i_layer]
    picName = "Baseline_vs_PosEmb_NoSkipCo{}_Layer{}_Seqlen{}_Depth{}_Hidden{}_convexhull_{}.png".format(sample_i,i_layer,seqlen,depth,dim,"SquareM" if use_squareM else "singular")
    m1_new = np.matmul(m1.transpose(1,0),m1)
    m2_new = np.matmul(m2.transpose(1,0),m2)
    print(m1.shape)
    print(m1_new.shape)
    if use_squareM:
        m1 = m1_new
        m2 = m2_new
        sv1, _ =  np.linalg.eig(m1)
        sv2, _ =  np.linalg.eig(m2)
    else:
        sv1, _ = singular_spectrum(m1)
        sv2, _ = singular_spectrum(m2)
    rank1 = np.linalg.matrix_rank(m1)
    rank2 = np.linalg.matrix_rank(m2)
    print("rank withSC: %d rank woSC: %d"%(rank1,rank2))
    draw_cdf(arr=[sv1,sv2],sample_i = sample_i,epoch="eval", picName=picName, auc1=auc1, auc2=auc2,dataset = dataset)
