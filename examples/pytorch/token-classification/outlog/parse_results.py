import ast
import os
import json
import re

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for apply_exrank in ["-add_last_afterln","-add_last_beforeln","_None"]:
        cwd = "/home/hanqiyan/repGeo/transformers/tokenUni/examples/pytorch/token-classification/outlog"
        for lnv in ["-soft_expand","baseline"]:
            path = os.path.join(cwd,"0701"+model+apply_exrank+lnv+".out")
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
    