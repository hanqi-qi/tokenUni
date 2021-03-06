export model_name=$3
CUDA_VISIBLE_DEVICES="0" python run_qa_no_trainer.py \
  --model_name_or_path $model_name \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --pad_to_max_length \
  --doc_stride 128 \
  --apply_exrank $1 \
  --lnv $2 \
  --spectral_norm True \
  --ifmask False \
  --exrank_nonlinear relu \
  --vis_step 500 \
  --output_dir /home/hanq1warwick/Data/rank_nips/tmp/debug_squad/$model_name/$2 \
  --seed 2021 \
  # --output_dir /home/hanq1warwick/Data/rank_nips/tmp/debug_squad/$model_name/$2
