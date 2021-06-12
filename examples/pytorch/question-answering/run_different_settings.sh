for lnv in "soft_expand_beta" "average" "linear"
do 
    # echo $apply_exrank
    for ifmask in "True"
    do
        for model_name in "bert-base-uncased" "distilbert-base-uncased" "albert-base-v1" "roberta-base"
        do
            bash run_qa_no_trainer.sh "add_every4" $lnv $ifmask $model_name>"./outlog/${model_name}_add_every4-${lnv}-${ifmask}.out" 2>&1
        done
    done
done

for model_name in "roberta-base" "bert-base-uncased" "distilbert-base-uncased" "albert-base-v1" 
do
    bash run_qa_no_trainer.sh "None" "origin" "False" $model_name >"./outlog/${model_name}_None-baseline.out" 2>&1

    # bash run_qa_no_trainer.sh "replace_all" "soft_expand_beta" "True" $model_name >"./outlog/${model_name}-soft_expand_beta-replace_all.out" 2>&1

done

