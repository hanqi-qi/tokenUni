for model_name in "bert-base-uncased"
do
    for TASK_NAME in "rte"
    do 
        bash run_glue_baselines.sh "None" "origin" $TASK_NAME $model_name >"./outlog/0606${model_name}_${TASK_NAME}_None_baseline.out" 2>&1
    done
done
