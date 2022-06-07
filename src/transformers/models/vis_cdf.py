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
from sklearn import base, preprocessing
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import torch.nn.functional as F

def singular_spectrum(W, norm=False): 
    if norm:
        W = W/np.trace(W)
    M = np.min(W.shape)
    svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
    svd.fit(W) 
    svals = svd.singular_values_
    svecs = svd.components_
    return svals, svecs

def multiCDF():
    average_batch = True
<<<<<<< HEAD
    dataset_name = "rte"
    model_type = "albert-base-v1"
    #pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/".format(dataset_name,model_type)
    pic_dir = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/pic/".format(dataset_name,model_type)
=======
    dataset_name = "mrpc"
    model_type = "albert-base-v1"
    pic_dir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/pic/".format(dataset_name,model_type)
    # pic_dir = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/pic/".format(dataset_name,model_type)
>>>>>>> 2f31e9d9d9bf8ee0afecfb104fd91e052b75d82d
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    fig,ax = plt.subplots(figsize=(8, 5))
    colors={"origin":["violet","black","cyan","peru","red","violet","blue","tomato"],
    "soft_expand":["lightcoral","violet","peru","red","tomato"]}
    linestyle={"origin":"solid","soft_expand":"solid"}
    # markers = {"origin":"*","soft_expand":None}
    # #draw baseline cdf
<<<<<<< HEAD
    #basicDataPath = datapath = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,"origin","2")
    #basicDataPath = datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,"origin","2")
    #basicData = np.load(basicDataPath)
    #sv=[]
    #if average_batch:
        #sv = []
        #for sample_i in range(basicData.shape[0]):
            #sv_batch, _ = singular_spectrum(basicData[sample_i,6,:,:])
            #sv.extend(sv_batch)
    #sv, _ = singular_spectrum(basicData[sample_i,6,:,:])
    #arr_1_edf = np.arange(1, 1+len(sv), 1)/len(sv)
    #arr_1_sorted = np.sort(sv/max(sv))#normalize to 1
    #plt.plot(arr_1_sorted, arr_1_edf, label='W.o.Layer6',color = "red",linewidth=1.5,linestyle="dotted")
    for lnv in ["soft_expand"]:
        for epoch in ["2"]:
            if lnv == "origin":
                #datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,lnv,epoch)
            	datapath = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,lnv,epoch)
            elif lnv == "soft_expand":
                #datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_add_last_beforeln.npy".format(dataset_name,model_type,lnv,epoch)
                datapath = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/weight/{}/Epoch{}_add_last_beforeln.npy".format(dataset_name,model_type,lnv,epoch)
=======
    # basicDataPath = datapath = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,"origin","2")
    # basicDataPath = datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,"origin","2")
    # basicData = np.load(basicDataPath)
    # sv=[]
    # if average_batch:
    #     sv = []
    #     for sample_i in range(basicData.shape[0]):
    #         sv_batch, _ = singular_spectrum(basicData[sample_i,12,:,:])
    #         sv.extend(sv_batch)
    # sv, _ = singular_spectrum(basicData[sample_i,12,:,:])
    # arr_1_edf = np.arange(1, 1+len(sv), 1)/len(sv)
    # arr_1_sorted = np.sort(sv/max(sv))#normalize to 1
    # plt.plot(arr_1_sorted, arr_1_edf, label='W.o.Layer12',color = "red",linewidth=1.5,linestyle="dotted")
    for lnv in ["soft_expand"]:
        for epoch in ["100"]:
            if lnv == "origin":
                datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,lnv,epoch)
                # datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_None.npy".format(dataset_name,model_type,lnv,epoch)
            elif lnv == "soft_expand":
                datapath = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/{}/weight/{}/Epoch{}_add_last_beforeln.npy".format(dataset_name,model_type,lnv,epoch)
                # datapath = "/home/hanq1warwick/Data/rank_nips/eigenout/{}/{}/weight/{}/Epoch{}_add_last_beforeln.npy".format(dataset_name,model_type,lnv,epoch)
>>>>>>> 2f31e9d9d9bf8ee0afecfb104fd91e052b75d82d
            basicData = np.load(datapath)#[bs,layer,sample_i,dim]
            for layer_i in range(basicData.shape[1]):
                if average_batch:
                    sv = []
                    for sample_i in range(basicData.shape[0]):
                        sv_batch, _ = singular_spectrum(basicData[sample_i,layer_i,:,:])
                        sv.extend(sv_batch)
                    if layer_i%3==0 and layer_i!=0:
                        arr_1_edf = np.arange(1, 1+len(sv), 1)/len(sv)
                        arr_1_sorted = np.sort(sv/max(sv))
                        plt.plot(arr_1_sorted, arr_1_edf, label='Layer%d'%(layer_i),color = colors["origin"][int(layer_i/3)],linewidth=1.5,linestyle=linestyle["origin"])
    plt.legend(prop={'size': 16})
    # plt.title("CDF for %s Task on %s" %(dataset_name,model_type))
    plt.ylabel("F(x)",size=16)
    plt.xlabel('x',size=16)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    child_dir = "0715{}_{}LastEpoch_{}every3Layer_cdf_average_nocompare.pdf".format(model_type,dataset_name,epoch)
    pic_name = os.path.join(pic_dir,child_dir)
    plt.savefig(pic_name,dpi=500)


<<<<<<< HEAD
multiCDF()
=======
multiCDF()
>>>>>>> 2f31e9d9d9bf8ee0afecfb104fd91e052b75d82d
