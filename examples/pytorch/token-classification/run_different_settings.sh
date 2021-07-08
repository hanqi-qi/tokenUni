for lnv in "soft_expand"
do
    for apply_exrank in "add_last_afterln" "add_last_beforeln"
    do
        for model_name in "bert-base-uncased" "albert-base-v1"
        do
            for decay_alpha in "-0.2" "-0.5" "-0.8"
            do
                for alpha_lr in "2e-3" "2e-5"
                do
                    bash run_ner_no_trainer.sh $apply_exrank $lnv $model_name $decay_alpha $alpha_lr>"./outlog/0706${model_name}_${apply_exrank}_${lnv}_Initalpha${decay_alpha}_alphaLr${alpha_lr}.out" 2>&1
                done
            done
        done
    done
done

# for model_name in "bert-base-uncased"
# do
#     for TASK_NAME in "sst2" "qnli"#"rte" "cola" "mrpc" 
#     do 
#         bash run_glue_no_trainer.sh "None" "origin" $TASK_NAME $model_name >"./outlog/0705${model_name}_${TASK_NAME}_None_baseline.out" 2>&1
#     done
# done

# for model_name in "albert-base-v1" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"
# do
#     bash run_ner_no_trainer.sh "None" "origin" $model_name >"./outlog/0703${model_name}_Nonebaseline_alphaN01.out" 2>&1
# done

