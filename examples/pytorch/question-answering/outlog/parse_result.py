import ast
import os
import json
import re

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for lnv in ["average","linear","soft_expand_beta"]:
        cwd = "/home/hanqiyan/transformers/examples/pytorch/question-answering/outlog"
        path = os.path.join(cwd,model+"_add_last-"+lnv+"-True.out")
        if os.path.exists(path):
            print(path)
            input_content = open(path).readlines()
            for line in input_content:
                if "Evaluation metrics: " in line.strip():
                    result = line.split("Evaluation metrics:")[-1]
                    print(result)
                    break
    