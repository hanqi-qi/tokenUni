for model_name in "bert-base-uncased" "roberta-base" "albert-base-v1"
do
    bash test.sh $model_name >"whitebert_real"+${model_name}.out 2>&1&
done