import ast
import os
import json
import re

for model in ["albert-base-v1","distilbert-base-uncased"]:#"albert-base-v1","roberta-base",
    for apply_exrank in ["_add_last_afterln","_add_last_beforeln"]:
        for decay_alpha in ["-0.2","-0.5","-0.8"]:
            for alpha_lr in ["2e-3","2e-5"]:
                cwd = "/home/hanqiyan/repGeo/transformers/tokenUni/examples/pytorch/token-classification/outlog"
                path = os.path.join(cwd,"0706"+model+apply_exrank+"_soft_expand"+"_Initalpha"+decay_alpha+"_alphaLr"+alpha_lr+".out")
                if os.path.exists(path):
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
    