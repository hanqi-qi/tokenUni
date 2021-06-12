import ast
import os
import json
import re

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for lnv in ["average-True","linear-True","soft_expand_beta-True","baseline"]:
        for apply_exrank in ["replace_last","add_every4","add_last","None"]:
            cwd = "/home/hanqiyan/transformers/examples/pytorch/language-modeling/outlog"
            path = os.path.join(cwd,model+"_"+apply_exrank+"-"+lnv+".out")
            if os.path.exists(path):
                print(path)
                input_content = open(path).readlines()
                for line in input_content:
                    if "epoch 2: perplexity:" in line.strip():
                        result = line.split("epoch 2: perplexity:")[-1]
                        print(result)
                        break
    