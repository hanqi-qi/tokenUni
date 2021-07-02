for lnv in "soft_transform"
do
    for apply_exrank in "add_last_afterln" "add_last_beforeln"
    do
        for model_name in "albert-base-v1" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"
        do
            for TASK_NAME in "rte" "cola" "mrpc"
            do
                bash run_glue_no_trainer.sh $apply_exrank $lnv $TASK_NAME $model_name>"./outlog/0702${model_name}_${TASK_NAME}_${apply_exrank}_${lnv}.out" 2>&1
            done
        done
    done
done

for model_name in "albert-base-v1" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"
do
    for TASK_NAME in "rte" "cola" "mrpc"
    do 
        bash run_glue_no_trainer.sh "None" "origin" $TASK_NAME $model_name >"./outlog/0702${model_name}_${TASK_NAME}_None_baseline.out" 2>&1
    done
done