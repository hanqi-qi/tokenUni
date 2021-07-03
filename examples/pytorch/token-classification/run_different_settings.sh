for lnv in "soft_transform"
do 
    for model_name in "albert-base-v1" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"
    do
        for apply_exrank in "add_last_beforeln" "add_last_afterln"
        do
            bash run_ner_no_trainer.sh $apply_exrank $lnv $model_name >"./outlog/0703${model_name}-${apply_exrank}-${lnv}_alphaN01.out" 2>&1
        done
    done
done

for model_name in "albert-base-v1" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"
do
    bash run_ner_no_trainer.sh "None" "origin" $model_name >"./outlog/0703${model_name}_Nonebaseline_alphaN01.out" 2>&1
done

