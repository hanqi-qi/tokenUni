# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export model_name=$4

CUDA_VISIBLE_DEVICES="0" python run_ner_no_trainer.py \
  --model_name_or_path $4 \
  --dataset_name conll2003 \
  --output_dir /mnt/Data3/hanqiyan/rank_transformers/tmp/test-ner/$4/$2 \
  --pad_to_max_length \
  --task_name ner \
  --return_entity_level_metrics \
  --lnv $2 \
  --apply_exrank $1 \
  --spectral_norm True \
  --exrank_nonlinear relu \
  --vis_step 500 \
  --ifmask $3 \
  --seed 2021 \
