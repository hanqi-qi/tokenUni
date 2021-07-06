for lnv in "soft_transform"
do 
    for apply_exrank in "add_last_afterln" "add_last_beforeln"
    do
        for model_name in "bert-base-uncased" "distilbert-base-uncased" "albert-base-v1" "roberta-base"
        do
            bash run_mlm_no_trainer.sh $apply_exrank $lnv $model_name>"./outlog/0704${model_name}_${apply_exrank}-${lnv}_alphaN01.out" 2>&1
        done
    done
done

for model_name in "roberta-base" "bert-base-uncased" "distilbert-base-uncased" "albert-base-v1" 
do
    bash run_mlm_no_trainer.sh "None" "origin" $model_name >"./outlog/0702${model_name}_None-baseline_alphaN01.out" 2>&1$
done

