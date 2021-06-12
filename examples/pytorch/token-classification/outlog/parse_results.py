import ast
import os
import json
import re

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for lnv in ["average","linear","soft_expand_beta"]:
        cwd = "/home/hanqiyan/transformers/examples/pytorch/token-classification/outlog"
        path = os.path.join(cwd,model+"_add_every4-"+lnv+"-True.out")
        if os.path.exists(path):
            # print(path)
            input_content = open(path).readlines()
            for line in input_content:
                if line.strip().startswith("epoch 2:"):
                    ll = line[9:].strip().replace("\'","\"")
                    result_dict=json.loads(ll)
                    print(path)
                    for key in ["LOC_f1","MISC_f1","ORG_f1","overall_precision","overall_recall","overall_f1"]:
                        if key in result_dict.keys():
                            print(key,result_dict[key])
                    break
    