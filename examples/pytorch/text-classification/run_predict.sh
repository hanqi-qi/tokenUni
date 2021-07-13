export TASK_NAME=$1
export model_name=$2

CUDA_VISIBLE_DEVICES="3" python run_glue.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --do_train \
  --do_eval \
  --do_predict \
  --output_dir /mnt/Data3/hanqiyan/rank_transformers/tmp/predict_glue0712/$model_name/$TASK_NAME/