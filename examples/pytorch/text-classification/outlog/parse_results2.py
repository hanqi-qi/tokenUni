import os

for model in ["bert-base-uncased","distilbert-base-uncased","albert-base-v1","roberta-base"]:
    for task in ["_qnli","rte" "wnli","_cola","_mrpc","_sst2"]:
        for lnv in ["_soft_expand","_baseline"]:
            for apply_exrank in ["_None"]:
                cwd = "/home/hanq1warwick/tokenUni/examples/pytorch/text-classification/outlog"
                path = os.path.join(cwd,"0708"+model+task+apply_exrank+lnv+".out")
                if os.path.exists(path):
                    print(path)
                    input_content = open(path).readlines()
                    for line in input_content:
                        if "epoch 2" in line:
                            result = line.split("epoch 2:")[-1]
                            print(result)
                            break
