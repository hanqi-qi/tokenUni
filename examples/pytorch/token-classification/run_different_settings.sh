for lnv in "soft_expand"
do
    for apply_exrank in "add_last_afterln" "add_last_beforeln"
    do
        for model_name in "distilbert-base-uncased" "roberta-base"
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
