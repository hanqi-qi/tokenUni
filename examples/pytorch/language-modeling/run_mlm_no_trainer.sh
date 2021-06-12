export  model_name=$4
CUDA_VISIBLE_DEVICES="0" python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path $4 \
    --output_dir /mnt/Data3/hanqiyan/rank_transformers/tmp/test-mlm/$4/$2 \
    --apply_exrank $1 \
    --lnv $2 \
    --spectral_norm True \
    --exrank_nonlinear relu \
    --vis_step 20 \
    --ifmask $3 \