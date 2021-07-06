import ast
import os
import json
import re

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for lnv in ["-soft_transform","-baseline"]:
        for apply_exrank in ["_add_last_afterln","_add_last_beforeln","_None"]:
            cwd = "/home/hanq1warwick/tokenUni/examples/pytorch/language-modeling/outlog"
            path = os.path.join(cwd,"0704"+model+apply_exrank+lnv+"_alphaN01.out")
            if os.path.exists(path):
                print(path)
                input_content = open(path).readlines()
                for line in input_content:
                    if "epoch 2: perplexity:" in line.strip():
                        result = line.split("epoch 2: perplexity:")[-1]
                        print(result)
                        break
    
