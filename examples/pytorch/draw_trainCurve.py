import numpy as np
import matplotlib.pyplot as plt
import os

def lossCurve(loss_list,picname,sampleRate=None):
    # lossNum = len(loss_list)
    line_color = ["red","black","green","blue","yellow"]
    label_list = ['Origin','Average','Linear','Trans']
    for i, loss in enumerate(loss_list):
        if sampleRate:
            sampleId = np.arange[0,len(loss),int(len(loss)/sampleRate)]
            y = loss[sampleId]
            x = range(len(y))
            plt.plot(x,y,color =line_color[i],label=label_list[i])
    plt.legend()
    plt.savefig(picname)

parent_dir = "/mnt/Data3/hanqiyan/rank_transformers/tmp/test-ner/bert-base-uncased/"
ori_loss = np.loadtxt(os.path.join(parent_dir,"origin/origin_None.txt"))
ave_loss =  np.loadtxt(os.path.join(parent_dir,"average/average_add_every4.txt"))
linear_loss = np.loadtxt(os.path.join(parent_dir,"linear/linear_add_every4.txt"))
trans_loss = np.loadtxt(os.path.join(parent_dir,"soft_expand_beta/soft_expand_beta_add_every4.txt"))
loss_list = [ori_loss,ave_loss,linear_loss,trans_loss]
picname = os.path.join(parent_dir,"nerloss.pdf")
lossCurve(loss_list,picname,sampleRate=100)