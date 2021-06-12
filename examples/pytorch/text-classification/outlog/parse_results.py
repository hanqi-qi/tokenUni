import os

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for task in ["cola","mrpc","mnli","qnli","rte"]:
        for lnv in ["average","linear","soft_expand_beta-True","soft_expand-True","soft_expand_False","soft_expand_beta-False","baseline"]:
            for apply_exrank in ["_replace_last","_add_last","_add_every4","_None"]:
                cwd = "/home/hanqiyan/transformers/examples/pytorch/text-classification/outlog"
                path = os.path.join(cwd,model+"_"+task+apply_exrank+"-"+lnv+"0525.out")
                if os.path.exists(path):
                    print(path)
                    input_content = open(path).readlines()
                    for line in input_content:
                        if "epoch 2" in line:
                            result = line.split("epoch 2:")[-1]
                            print(result)
                            break
    