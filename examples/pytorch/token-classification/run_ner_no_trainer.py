#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""
from re import T
import textwrap
from sklearn.manifold import TSNE
from adjustText import adjust_text
import argparse
import logging
import math
import os
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
import datetime

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm.utils import _text_width

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from transformers import BertConfig
from transformers import DistilBertConfig
from transformers import AlbertConfig
from transformers import RobertaConfig

from transformers.models import vis_tools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy import stats
import shutil

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
def singular_spectrum(W, norm=False): 
    if norm:
        W = W/np.trace(W)
    M = np.min(W.shape)
    svd = TruncatedSVD(n_components=M-1, n_iter=7, random_state=10)
    svd.fit(W) 
    svals = svd.singular_values_
    svecs = svd.components_
    return svals, svecs

def tokenSimi(tokens_matrix,seqlen,nopad=False):
    """calculate the average cosine similarity,with/without normalization"""
    simi = []
    if nopad:
        l = seqlen
    else:
        l = tokens_matrix.shape[0]
    for i in range(l):
        for j in range(l):
            if i!=j:
                simi.append(dot(tokens_matrix[i],tokens_matrix[j])/(norm(tokens_matrix[i])*norm(tokens_matrix[j])))
    return sum(simi)/len(simi)

def vis_eigenValue(tokens_matrix,i_layer,batch_id,pic_dir,epoch,rank,simiValue):
    sv, _ = singular_spectrum(tokens_matrix)
    sv_stats= stats.describe(sv/np.expand_dims(np.max(sv),-1))
    sv_stats_ori= stats.describe(sv)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.abs(sv), 50)
    ax.set_yscale('log')
    plt.title("Epoch {} layer {} ".format(epoch,i_layer))
    ax.text(ax.get_xlim()[1]*0.8,0.5*ax.get_ylim()[1],"mean:"+str("%.2f"%sv_stats.mean)+"\n"+"maxmin:("+str("%.2f"%sv_stats.minmax[0])+","+str("%.2f"%sv_stats.minmax[1])+")"+"\n"+"variance:"+str("%.2f"%sv_stats.variance)+"\n"+"skewness:"+str("%.2f"%sv_stats.skewness)+"\n"+"kurtosis:"+str("%.2f"%sv_stats.kurtosis)+"\n"+"rank:"+str("%d"%rank)+"\n"+"token_CosSimi:%.2f"%simiValue)
    #add no norm statistics
    ax.text(ax.get_xlim()[1]*0.8,0.01*ax.get_ylim()[1],"mean:"+str("%.2f"%sv_stats_ori.mean)+"\n"+"maxmin:("+str("%.2f"%sv_stats_ori.minmax[0])+","+str("%.2f"%sv_stats_ori.minmax[1])+")"+"\n"+"variance:"+str("%.2f"%sv_stats_ori.variance)+"\n"+"skewness:"+str("%.2f"%sv_stats_ori.skewness)+"\n"+"kurtosis:"+str("%.2f"%sv_stats_ori.kurtosis)+"\n"+"rank:"+str("%d"%rank))
    plt.savefig(os.path.join(pic_dir,"HistEpoch_{}BatchId_{}Layer_{}.png".format(epoch,batch_id,i_layer)))

def vis_tokenUni(hidden_states,input_ids,tokenizer,pic_dir,ifpca,args,epoch):
    # time_start = time.time()
    hidden_states = torch.stack(hidden_states).permute(1,0,2,3)
    batch_id = 0
    for batch_id in range(hidden_states.shape[0]):
        tokens_layers =  hidden_states[batch_id].cpu().detach().numpy()
        for i_layer in range(tokens_layers.shape[0]):
            tokens_matrix = tokens_layers[i_layer]
            
            if args.model_name_or_path == "bert-base-uncased":
                endToken = "[SEP]"
            elif args.model_name_or_path == "roberta-base":
                    endToken = "</s>"
            text_tokens = tokenizer.decode(input_ids[batch_id]).split(endToken)[0].split(" ")
            #only visualize the sequence longer than 20
            if len(text_tokens)>20:
                rank = np.linalg.matrix_rank(tokens_matrix)
                simiValue=tokenSimi(tokens_matrix=tokens_matrix,seqlen=len(text_tokens),nopad=False)
                #visualize the hist and cdf of the current token sequence.
                vis_eigenValue(tokens_matrix,i_layer,batch_id,pic_dir,epoch,rank,simiValue)
                if ifpca:
                    pca_50 = PCA(n_components=50)
                    pca_result_50 = pca_50.fit_transform(tokens_matrix)
                    tokens = pca_result_50
                tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                tsne_results = tsne.fit_transform(tokens)
                #calculate the token_similarity. nopad==True:clip the [PAD]
                
                d = {'tsne-2d-one': tsne_results[:,0][:len(text_tokens)], 'tsne-2d-two':tsne_results[:,1][:len(text_tokens)],'point_name':text_tokens}
                df_subset = pd.DataFrame(data=d)
                fig,ax = plt.subplots(figsize=(6, 6))
                df_subset.plot.scatter(x="tsne-2d-one",y="tsne-2d-two",s=25,ax=ax)
                texts = []
                for x, y, s in zip(df_subset["tsne-2d-one"], df_subset["tsne-2d-two"], df_subset["point_name"]):
                    if len(s)>0:
                        texts.append(plt.text(x, y, s))
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
                plt.title("{}_{}_layer{}Epoch_{}".format(args.model_name_or_path,args.task_name,i_layer,epoch))#adjust the token annotation
                # sentence = " ".join(text_tokens)
                # multi_sen=textwrap.wrap(sentence,width=75)
                # new_sen = "\n".join(multi_sen)
                # plt.xlabel(new_sen)
                picname = os.path.join(pic_dir,"tokenDisEpoch_{}BatchId_{}_Layer_{}0522.png".format(epoch,batch_id,i_layer))
                plt.savefig(picname)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    #TODO(yhq): add arguments for customized layer
    parser.add_argument(
        "--apply_exrank",
        type=str,
        default=None,
        help="different methods to apply our exrank layer, just add or replace original layernorm, at the last layer or for each layer"
    )
    parser.add_argument(
        "--lnv",
        type=str,
        default=None,
        help="different versions of expand exrank mechanisms."
    )
    parser.add_argument(
        "--spectral_norm",
        type=bool,
        default=False,
        help="if use the spectral norm."
    )
    parser.add_argument(
        "--exrank_nonlinear",
        type=str,
        default="relu",
        help="different nonlinear functions to the exrank layer."
    )
    parser.add_argument(
        "--vis_step",
        type=int,
        default="1000",
        help="save the intermediate output each vis_epoch"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--mask_hidden_dim",
        type=int,
        default = 300,
        help="the hidden dimension of the eigenvalue mask",
    )
    parser.add_argument(
        "--ifmask",
        type=bool,
        default = False,
        help="if mask the eigenvalue",
    )
    
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = f"{args.task_name}_tags" if f"{args.task_name}_tags" in column_names else column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    #TODO(yhq): idk why the label number is not compatible with the configuration
    # num_labels = 14
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    #TODO(yhq0514):loading the pretrained model
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:#
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True,add_prefix_space=True )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True,add_prefix_space=True )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    #TODO(yhq): add customized arguments and use the default ones in Config file.
    id2label_dict= {
        "0": "LABEL_0",
        "1": "LABEL_1",
        "2": "LABEL_2",
        "3": "LABEL_3",
        "4": "LABEL_4",
        "5": "LABEL_5",
        "6": "LABEL_6",
        "7": "LABEL_7",
        "8": "LABEL_8",
        "9": "LABEL_9",
        "10": "LABEL_10",
        "11": "LABEL_11",
        "12": "LABEL_12",
        "13": "LABEL_13"
    }
    label2id_dict= {
        "LABEL_0": 0,
        "LABEL_1": 1,
        "LABEL_2": 2,
        "LABEL_3": 3,
        "LABEL_4": 4,
        "LABEL_5": 5,
        "LABEL_6": 6,
        "LABEL_7": 7,
        "LABEL_8": 8,
        "LABEL_9": 9,
        "LABEL_10": 10, 
        "LABEL_11": 11, 
        "LABEL_12": 12, 
        "LABEL_13": 13
    }
    if args.model_name_or_path=="albert-base-v1":
        config = AlbertConfig(
            lnv = args.lnv,
            max_seq_length = args.max_length,
            spectral_norm = args.spectral_norm,
            exrank_nonlinear = args.exrank_nonlinear,
            apply_exrank = args.apply_exrank,
            return_dict = True,
            hidden_size = 768,#add for base model
            num_attention_heads=12,#add for base model
            intermediate_size=3072,#add for base model
            batch_size_yhq  = args.per_device_train_batch_size,
            ifmask = args.ifmask,
            mask_hidden_dim = args.mask_hidden_dim,
            output_hidden_states = True,
            id2label = id2label_dict,
            label2id = label2id_dict,
        )
    elif args.model_name_or_path=="distilbert-base-uncased":
            config = DistilBertConfig(
            lnv = args.lnv,
            max_seq_length = args.max_length,
            spectral_norm = args.spectral_norm,
            exrank_nonlinear = args.exrank_nonlinear,
            apply_exrank = args.apply_exrank,
            return_dict = True,
            batch_size_yhq  = args.per_device_train_batch_size,
            ifmask = args.ifmask,
            mask_hidden_dim = args.mask_hidden_dim,
            output_hidden_states = True,
            id2label=id2label_dict,
            label2id=label2id_dict,
        )
    elif args.model_name_or_path == "roberta-base":
        #initiate the config with super class bert.
        config = RobertaConfig(
            lnv = args.lnv,
            max_seq_length = args.max_length,
            spectral_norm = args.spectral_norm,
            exrank_nonlinear = args.exrank_nonlinear,
            apply_exrank = args.apply_exrank,
            return_dict = True,
            batch_size_yhq  = args.per_device_train_batch_size,
            ifmask = args.ifmask,
            mask_hidden_dim = args.mask_hidden_dim,
            output_hidden_states = True,
            id2label=id2label_dict,
            label2id=label2id_dict,
            intermediate_size = 3072,
            hidden_size= 768,
            max_position_embeddings = 514,
            type_vocab_size=1,
            vocab_size=50265,
        )
    elif args.model_name_or_path == "bert-base-uncased":
        config = BertConfig(
            lnv = args.lnv,
            max_seq_length = args.max_length,
            spectral_norm = args.spectral_norm,
            exrank_nonlinear = args.exrank_nonlinear,
            apply_exrank = args.apply_exrank,
            return_dict = True,
            batch_size_yhq  = args.per_device_train_batch_size,
            ifmask = args.ifmask,
            mask_hidden_dim = args.mask_hidden_dim,
            output_hidden_states = True,
            id2label=id2label_dict,
            label2id=label2id_dict,
        )
    else:
        print("Invalid model type!!!")

    print(config)
    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the raw_datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    #TODO(yhq): add soft_expand layer parameters as no_decay group. A approximate frozen version.
    # no_decay = ["bias", "LayerNorm.weight"] #subtring in the parameter's name
    #TODO(yhq): larger learning rate for the alpha in soft-expand function.
    no_decay = ["bias", "LayerNorm.weight","exrank_layer"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    metric = load_metric("seqeval")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    #!Train
    picdir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/conll2003/token_uni/{}/{}".format(args.model_name_or_path,args.lnv)
    if os.path.isdir(picdir):
        shutil.rmtree(picdir)
    if not os.path.isdir(picdir):
        os.makedirs(picdir)
    time_list = []
    loss_train,loss_test = {},{}
    loss_x = 0
    for _ in range(1):
        for epoch in range(args.num_train_epochs):
            start_time = datetime.datetime.now()
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)#
                # vis_tokenUni(outputs.hidden_states,batch["input_ids"],tokenizer,picdir,ifpca,args,step)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                loss_train[str(loss_x)] = loss.item()
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                if completed_steps >= args.max_train_steps:
                    break
                # if step%args.vis_step==0:
                    # vis_tokenUni(outputs.hidden_states,batch["input_ids"],tokenizer,picdir,ifpca,args,step)
            # vis_tools.save_matrix(outputs.hidden_states,epoch,args,mode='train',timestamps='new')
                #TODO(yhq):save the intermediate hidden_states every vis_epoch
                if step%args.vis_step ==0:
                    hidden_states_layers = torch.stack(outputs.hidden_states).permute(1,0,2,3)
                #     #TODO(yhq0509): see the soft_expand function effects. draw the hist, and statistics after NORM
                    # old_hidden_states = torch.stack(outputs.old_hidden_states).permute(1,0,2,3)
                    # vis_tools.save_matrix(old_hidden_states,step,args,mode='train',timestamp='old2')
                    vis_tools.save_matrix(hidden_states_layers,step,args,mode="train",timestamp='new')
            end_time = datetime.datetime.now()
            time_list.append((end_time-start_time).seconds)
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)
                #
                preds, refs = get_labels(predictions_gathered, labels_gathered)
                metric.add_batch(
                    predictions=preds,
                    references=refs,
                ) 
            eval_metric = compute_metrics()
            accelerator.print(f"epoch {epoch}:", eval_metric)
    lossfile = os.path.join(args.output_dir,"{}_{}.txt".format(args.lnv,args.apply_exrank))
    np.savetxt(lossfile,np.array(list(loss_train.values())))
    print("****Cost Time Every Epoch****")
    print(sum(time_list)/len(time_list))
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    
if __name__ == "__main__":
    main()
