for lnv in "soft_expand"
do
    for apply_exrank in "add_last_beforeln"
    do
        for model_name in "roberta-base"
        do
            for TASK_NAME in "mrpc" 
            do
                for decay_alpha in "-0.1" 
                do
                    for alpha_lr in "2e-5"
                    do
                        bash run_glue_no_trainer.sh $apply_exrank $lnv $TASK_NAME $model_name $decay_alpha $alpha_lr>"./outlog/0422_addRoberta_${model_name}_${TASK_NAME}_${apply_exrank}_${lnv}.out" 2>&1
                    done
                done
            done
        done
    done
done