export model_name=$3

CUDA_VISIBLE_DEVICES="0" python run_ner_no_trainer.py \
>>>>>>> 29efca0c426830ab06181f05c4ce94667ca24f83
  --model_name_or_path $3 \
  --dataset_name conll2003 \
  --output_dir /home/hanq1warwick/Data/rank_nips/tmp/test-ner/$3/$2 \
  --pad_to_max_length \
  --task_name ner \
  --return_entity_level_metrics \
  --lnv $2 \
  --apply_exrank $1 \
  --spectral_norm True \
  --exrank_nonlinear relu \
  --vis_step 500 \
  --ifmask False \
  --seed 2021 \
  --decay_alpha $4 \
  --alpha_lr $5 \
