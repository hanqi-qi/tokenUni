for lnv in "soft_expand"
do 
    # echo $apply_exrank
    for apply_exrank in "add_last_afterln" "add_last_beforeln"
    do
        for model_name in "roberta-base" "bert-base-uncased" "distilbert-base-uncased" "albert-base-v1" 
        do
            bash run_qa_no_trainer.sh $apply_exrank $lnv $model_name>"./outlog/0630${model_name}_${apply_exrank}-${lnv}-${ifmask}.out" 2>&1
        done
    done
done

for model_name in "roberta-base" "bert-base-uncased" "distilbert-base-uncased" "albert-base-v1" 
do
    bash run_qa_no_trainer.sh "None" "origin" $model_name >"./outlog/0630${model_name}_None-baseline.out" 2>&1
done

