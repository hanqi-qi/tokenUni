import ast
import os
import json
import re

# for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
for lnv in ["soft_expand"]:
    cwd = "/home/hanqiyan/test/WhiteningBERT/examples/evaluation/"
    path = os.path.join(cwd,"softexpand_neigh.out")
    if os.path.exists(path):
        print(path)
        input_content = open(path).readlines()
        for line in input_content:
            if line.strip().startswith("0."):
                ori_error,t_error = line.strip().split(" ")
                # structloss=(float(ll[0][7:-2])+float(ll[1][7:-2]))/2
                print(ori_error,t_error)
    