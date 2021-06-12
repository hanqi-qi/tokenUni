for lnv in "soft_expand"
do 
    for ifmask in "True"
    do
        for model_name in "albert-base-v1"
        do
            # bash run_ner_no_trainer.sh "add_last" $lnv $ifmask $model_name>"./outlog/${model_name}_add_last-${lnv}-${ifmask}0526.out" 2>&1
            bash run_ner_no_trainer.sh "replace_last" $lnv $ifmask $model_name>"./outlog/${model_name}_replace_last-${lnv}-${ifmask}0603night.out" 2>&1
        done
    done
done

for model_name in "albert-base-v1"
do
    bash run_ner_no_trainer.sh "None" "origin" "False" $model_name >"./outlog/${model_name}_None-baseline0603weight.out" 2>&1

    # bash run_ner_no_trainer.sh "add_all" "soft_expand_beta" "True" $model_name >"./outlog/${model_name}-soft_expand_beta-add_all.out" 2>&1

done

