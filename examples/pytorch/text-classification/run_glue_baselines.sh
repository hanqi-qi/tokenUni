export TASK_NAME=$2
export model_name="bert-base-uncased"

CUDA_VISIBLE_DEVICES="0" python run_glue_no_trainer.py \
  --model_name_or_path $model_name \
  --task_name $TASK_NAME \
  --dataset_name $TASK_NAME \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --output_dir /home/hanq1warwick/Data/rank_nip/tmp/debug_glue/$model_name/$TASK_NAME/$1 \
  --lnv $1 \
  --spectral_norm True \
  --exrank_nonlinear relu \
  --vis_step 50 \
  --ifmask False \
  --seed 2021 \
  --decay_alpha -0.2 \
  --alpha_lr 2e-5 \
