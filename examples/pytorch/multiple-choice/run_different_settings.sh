# for apply_exrank in "add_last" "replace_last" "replace_all"
# do 
#     # echo $apply_exrank
#     bash run_swag_no_trainer.sh $apply_exrank "exrank_gx" >"./outlog/ApplyExrank_${apply_exrank}-exrank_gx.out" 2>&1
# done;

bash run_swag_no_trainer.sh "None" "origin" >"./outlog/ApplyExrank_None-baseline.out" 2>&1
