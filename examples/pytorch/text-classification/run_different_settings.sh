for lnv in "soft_expand"
do
<<<<<<< HEAD
    for apply_exrank in "add_last_beforeln"
    do
        for model_name in "bert-base-uncased"
        do
            for TASK_NAME in "rte" "cola" "mrpc" "sst2" "qnli" 
            do
                for decay_alpha in "-0.2" 
                do
                    for alpha_lr in "2e-5"
                    do
                        bash run_glue_no_trainer.sh $apply_exrank $lnv $TASK_NAME $model_name $decay_alpha $alpha_lr>"./outlog/0711_test${model_name}_${TASK_NAME}_${apply_exrank}_${lnv}.out" 2>&1
=======
    for apply_exrank in "add_last_afterln" "add_last_beforeln"
    do
        for model_name in "distilbert-base-uncased" "albert-base-v1" "roberta-base"
        do
            for TASK_NAME in "rte" "cola" "mrpc" "sst2" "qnli" 
            do
                for decay_alpha in "-0.2" "-0.5" "-0.8"
                do
                    for alpha_lr in "2e-3" "2e-5"
                    do
                        bash run_glue_no_trainer.sh $apply_exrank $lnv $TASK_NAME $model_name $decay_alpha $alpha_lr>"./outlog/0706${model_name}_${TASK_NAME}_${apply_exrank}_${lnv}_Initalpha${decay_alpha}_alphaLr${alpha_lr}.out" 2>&1
>>>>>>> 4e412e387c418cac1cc74e8703126a23ddfac23a
                    done
                done
            done
        done
    done
done

<<<<<<< HEAD
for lnv in "origin"
=======
# for model_name in "bert-base-uncased"
# for lnv in "soft_expand"
# do
#     for apply_exrank in "add_last_afterln" "add_last_beforeln"
#     do
#         for model_name in "bert-base-uncased"
#         do
#             for TASK_NAME in "rte" "cola" "mrpc" "sst2" "qnli" 
#             do
#                 for decay_alpha in "-0.2" "-0.5" "-0.8"
#                 do
#                     for alpha_lr in "2e-3" "2e-5"
#                     do
#                         bash run_glue_no_trainer.sh $apply_exrank $lnv $TASK_NAME $model_name $decay_alpha $alpha_lr>"./outlog/0706${model_name}_${TASK_NAME}_${apply_exrank}_${lnv}_Initalpha${decay_alpha}_alphaLr${alpha_lr}.out" 2>&1
#                     done
#                 done
#             done
#         done
#     done
# done
for model_name in "bert-base-uncased" "albert-base-v1" "distilbert-base-uncased" "roberta-base"
>>>>>>> 4e412e387c418cac1cc74e8703126a23ddfac23a
do
    for apply_exrank in "None"
    do
        for model_name in "bert-base-uncased"
        do
            for TASK_NAME in "rte" "cola" "mrpc" "sst2" "qnli" 
            do
                for decay_alpha in "-0.2" 
                do
                    for alpha_lr in "2e-5"
                    do
                        bash run_glue_no_trainer.sh $apply_exrank $lnv $TASK_NAME $model_name $decay_alpha $alpha_lr>"./outlog/0711_test${model_name}_${TASK_NAME}_None_baseline.out" 2>&1
                    done
                done
            done
        done
    done
done
