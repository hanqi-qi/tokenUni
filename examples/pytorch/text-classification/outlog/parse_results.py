import os

for model in ["bert-base-uncased","distilbert-base-uncased"]:
    for task in ["_mrpc"]: #"rte" "wnli","_cola","_mrpc","_sst2","_qnli"
        for lnv in ["_soft_expand"]:
            for apply_exrank in ["_add_last_afterln","_add_last_beforeln","_None"]:
                for decay_alpha in ["-0.2","-0.5","-0.8"]:
                    for alpha_lr in ["2e-3","2e-5"]:
                        cwd = "/home/hanqiyan/repGeo/transformers/tokenUni/examples/pytorch/text-classification/outlog"
                        path = os.path.join(cwd,"0706"+model+task+apply_exrank+lnv+"_Initalpha"+decay_alpha+"_alphaLr"+alpha_lr+".out")
                        if os.path.exists(path):
                            print(path)
                            input_content = open(path).readlines()
                            for line in input_content:
                                if "epoch 2" in line:
                                    result = line.split("epoch 2:")[-1]
                                    print(result)
                                    break
