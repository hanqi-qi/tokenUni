export model_name=$1
CUDA_VISIBLE_DEVICES="3" python run_qa_no_trainer.py \
  --model_name_or_path $model_name \
  --dataset_name squad \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --pad_to_max_length \
  --doc_stride 128 \
  --output_dir /mnt/Data3/hanqiyan/rank_transformers/tmp/debug_squad/$model_name
  # --output_dir /home/hanq1warwick/Data/rank_nips/tmp/debug_squad/$model_name/$2