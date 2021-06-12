export TASK_NAME=$4
export model_name=$5

CUDA_VISIBLE_DEVICES="2" python run_glue_no_trainer.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --dataset_name $TASK_NAME \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --apply_exrank $1 \
  --output_dir /mnt/Data3/hanqiyan/rank_transformers/tmp/debug_glue/$model_name/$TASK_NAME/$2 \
  --lnv $2 \
  --spectral_norm True \
  --exrank_nonlinear relu \
  --vis_step 50 \
  --ifmask $3 \
  --seed 2021 \