for lnv in "soft_expand"
do
    for ifmask in "False"
    do
        for model_name in "distilbert-base-uncased"
        do
            for TASK_NAME in "mrpc" "cola" 
            do
                #bash run_glue_no_trainer.sh "add_last" $lnv $ifmask $TASK_NAME $model_name>"./outlog/${model_name}_${TASK_NAME}_add_last-${lnv}-${ifmask}0528.out" 2>&1
                bash run_glue_no_trainer.sh "replace_last" $lnv $ifmask $TASK_NAME $model_name>"./outlog/${model_name}_${TASK_NAME}-replace_last-${lnv}-${ifmask}0603night.out" 2>&1

            done
        done
    done
done

for model_name in "distilbert-base-uncased"
do
    for TASK_NAME in "mrpc" "cola"
    do 
        bash run_glue_no_trainer.sh "None" "origin" "False" $TASK_NAME $model_name >"./outlog/${model_name}_${TASK_NAME}_None-baseline0604.out" 2>&1
    done 

done