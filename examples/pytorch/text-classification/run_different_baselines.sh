for model_name in "albert-base-v1"
do
    for TASK_NAME in "sst2" "qnli" "rte" "cola" "mrpc" 
    do 
        bash run_glue_baselines.sh "None" "origin" $TASK_NAME $model_name >"./outlog/0708${model_name}_${TASK_NAME}_None_baseline.out" 2>&1
    done
done
