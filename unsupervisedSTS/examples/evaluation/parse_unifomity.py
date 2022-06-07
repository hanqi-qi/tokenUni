import ast
import os
import json
import re

# for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
for lnv in ["distilbert"]:
    cwd = "/home/hanqiyan/test/WhiteningBERT/examples/evaluation/"
    path = os.path.join(cwd,lnv+"_tsneUni.out")
    if os.path.exists(path):
        print(path)
        input_content = open(path).readlines()
        for line in input_content:
            if line.strip().startswith("origin"):
                ll = line.strip().split(" ")
                top1ev,rbfdis,tokenuni=float(ll[1][7:-1]),float(ll[2][7:-1]),float(ll[3][7:-1])
                print(top1ev,rbfdis,tokenuni)
    