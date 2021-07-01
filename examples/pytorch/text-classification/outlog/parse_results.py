import os

for model in ["albert-base-v1","roberta-base","bert-base-uncased","distilbert-base-uncased"]:
    for task in ["_cola","_mrpc","_mnli","_qnli","_rte"]:
        for lnv in ["_baseline","_soft_expand"]:
            for apply_exrank in ["_add_last_afterln","_add_last_beforeln","_None"]:
                cwd = "/home/hanqiyan/repGeo/transformers/tokenUni/examples/pytorch/text-classification/outlog"
                path = os.path.join(cwd,"0630_v2"+model+task+apply_exrank+lnv+".out")
                if os.path.exists(path):
                    print(path)
                    input_content = open(path).readlines()
                    for line in input_content:
                        if "epoch 2" in line:
                            result = line.split("epoch 2:")[-1]
                            print(result)
                            break
    