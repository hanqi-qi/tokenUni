import ast
import os
import json
import re

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for lnv in ["-baseline","-soft_expand-"]:
        for apply_exrank in ["_add_last_afterln","_add_last_beforeln","_None"]:
            cwd = "/home/hanq1warwick/tokenUni/examples/pytorch/question-answering/outlog"
            path = os.path.join(cwd,"0630"+model+apply_exrank+lnv+".out")
            if os.path.exists(path):
                print(path)
                input_content = open(path).readlines()
                for line in input_content:
                    if "Evaluation metrics: " in line.strip():
                        result = line.split("Evaluation metrics:")[-1]
                        print(result)
                        break
    
