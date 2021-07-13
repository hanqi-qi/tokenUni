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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import textwrap
import os
import random
from adjustText import adjust_text
import torch
import datetime
import numpy as np
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import sys
sys.path.pop()
sys.path.insert(0,"/home/hanqiyan/repGeo/transformers/tokenUni/src/")
import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
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

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
# def lossCurve(train_loss, test_loss,args):
#     x1 = train_loss.keys()
#     x2 = test_loss.keys()
#     y1 = train_loss.values()
#     y2 = test_loss.values()
#     plt.plot(x1,y1)
#     plt.plot(x2,y2)
#     parent_dir = args.output_dir
#     child_dir = "{}_{}.png".format(args.lnv,args.apply_exrank)
#     pic_name = os.path.join(parent_dir,child_dir)
#     plt.savefig(pic_name)
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

def vis_tokenUni(hidden_states,input_ids,labels,tokenizer,pic_dir,ifpca,args,epoch):
    # time_start = time.time()
    hidden_states = torch.stack(hidden_states).permute(1,0,2,3)
    batch_id = 0
    for batch_id in range(hidden_states.shape[0]):
        tokens_layers =  hidden_states[batch_id].cpu().detach().numpy()
        label_batch  = labels[batch_id].cpu().detach().numpy()
        for i_layer in range(tokens_layers.shape[0]):
            tokens_matrix = tokens_layers[i_layer]
            
            if args.model_name_or_path == "bert-base-uncased":
                endToken = "[SEP]"
            elif args.model_name_or_path == "roberta-base":
                    endToken = "</s>"
            split_token = "."
            sentence = tokenizer.decode(input_ids[batch_id]).split(endToken)[0]
            sent1_end = len(sentence.split(" "))
            # sent1_end = len(sent1.split(" "))
            text_tokens = " ".join((tokenizer.decode(input_ids[batch_id]).split(endToken))[:-1]).split(" ")
            #only visualize the sequence longer than 20
            if label_batch == 0:
                rank = np.linalg.matrix_rank(tokens_matrix)
                simiValue=tokenSimi(tokens_matrix=tokens_matrix,seqlen=len(text_tokens),nopad=False)
                #visualize the hist and cdf of the current token sequence.
                vis_eigenValue(tokens_matrix,i_layer,batch_id,pic_dir,epoch,rank,simiValue)
                if ifpca:
                    n_components = min(tokens_matrix.shape[0],tokens_matrix.shape[1])
                    pca_50 = PCA(n_components=n_components)
                    pca_result_50 = pca_50.fit_transform(tokens_matrix)
                    tokens = pca_result_50
                tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                tsne_results = tsne.fit_transform(tokens)
                #calculate the token_similarity. nopad==True:clip the [PAD]
                
                d = {'tsne-2d-one': tsne_results[:,0][:len(text_tokens)], 'tsne-2d-two':tsne_results[:,1][:len(text_tokens)],'point_name':text_tokens}
                df_subset = pd.DataFrame(data=d)
                fig,ax = plt.subplots(figsize=(8, 8))
                df_subset.plot.scatter(x="tsne-2d-one",y="tsne-2d-two",s=25,ax=ax)
                texts = []
                i = 0
                for x, y, s in zip(df_subset["tsne-2d-one"], df_subset["tsne-2d-two"], df_subset["point_name"]):
                    if len(s)>0:
                        if i<sent1_end:
                            texts.append(plt.text(x, y, s,color = "red",size=14))
                        else:
                            texts.append(plt.text(x, y, s,color = "blue",size=14))
                    i +=1
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
                plt.title("{}_{}_layer{}Epoch_{}".format(args.model_name_or_path,args.task_name,i_layer,epoch))#adjust the token annotation
                sentence = " ".join(text_tokens)
                multi_sen=textwrap.wrap(sentence,width=75)
                new_sen = "\n".join(multi_sen)
                plt.xlabel(new_sen)
                picname = os.path.join(pic_dir,"0529color_tokenDisEpoch_{}BatchId_{}_Layer_{}.png".format(epoch,batch_id,i_layer))
                plt.savefig(picname)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    #TODO(yhq): add arguments for customized layer
    parser.add_argument(
        "--apply_exrank",
        type=str,
        default=None,
        help="different methods to apply our exrank layer, just add or replace original layernorm, at the last layer or for each layer"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="add this argument for vis_tools"
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
        default="100",
        help="save the intermediate output each vis_epoch"
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
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
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
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
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
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
        default=2e-5,
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
    parser.add_argument(
        "--decay_alpha",
        type=float,
        default = -0.2,
        help="initial alpha value for soft_decay",
    )
    parser.add_argument(
        "--ifmask",
        type=bool,
        default = False,
        help="if mask the eigenvalue",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--alpha_lr", type=float, default=2e-3, help="learning rate the soft_decay function")
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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)

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
            decay_alpha = args.decay_alpha,
            output_hidden_states = True,
            # id2label = id2label_dict,
            # label2id = label2id_dict,
            num_labels = num_labels,
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
            decay_alpha = args.decay_alpha,
            output_hidden_states = True,
            # id2label=id2label_dict,
            num_labels = num_labels,
            # label2id=label2id_dict,
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
            decay_alpha = args.decay_alpha,
            output_hidden_states = True,
            # id2label=id2label_dict,
            # label2id=label2id_dict,
            intermediate_size = 3072,
            hidden_size= 768,
            max_position_embeddings = 514,
            type_vocab_size=1,
            vocab_size=50265,
        )
    elif args.model_name_or_path=="bert-base-uncased":
        config = BertConfig(
            lnv = args.lnv,
            max_seq_length = args.max_length,
            spectral_norm = args.spectral_norm,
            exrank_nonlinear = args.exrank_nonlinear,
            apply_exrank = args.apply_exrank,
            return_dict = True,
            batch_size_yhq  = args.per_device_train_batch_size,
            ifmask = args.ifmask,
            num_labels = num_labels,
            decay_alpha = args.decay_alpha,
            output_hidden_states = True,
            # id2label=id2label_dict,
            # label2id=label2id_dict,
        )
    else:
        print("Invalid model type!!!")

    print("After re-arguments:")
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    #TODO(yhq)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    #TODO(yhq0619):
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    predict_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    no_fix_weight = ["bert.pooler.dense.weight",'bert.pooler.dense.bias',"classifier.weight","classifier.bias","alpha"]
    # alpha_decay = ["alpha"]
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
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in alpha_decay)],
    #         "lr": args.learning_rate,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in alpha_decay)],
    #         "lr": args.alpha_lr,
    #     },
    # ]

    # optimizer = AdamW(optimizer_grouped_parameters)
    optimizer = AdamW(optimizer_grouped_parameters, lr = 2e-5)
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

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

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
    time_list = []
    loss_train,loss_test = {},{}
    loss_x = 0

    picdir = "/mnt/Data3/hanqiyan/rank_transformer/eigenout/{}/token_uni/{}/{}".format(args.task_name,args.model_name_or_path,args.lnv)
    if os.path.isdir(picdir):
        shutil.rmtree(picdir)
    if not os.path.isdir(picdir):
        os.makedirs(picdir)
    ifpca = True
    for epoch in range(args.num_train_epochs):
        start_time = datetime.datetime.now()
        model.train()
        for step, batch in enumerate(train_dataloader):
            loss_x +=1
            outputs = model(**batch)#original is batch
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss_train[str(loss_x)]=loss.item()
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
            #TODO(yhq):save the intermediate hidden_states every vis_epoch. only for the 
            #TODO(yhq): whitebert output
            if step%10000 ==0:
                if args.lnv=="soft_expand":
                    print(outputs.alpha.item())
                hidden_states_layers = torch.stack(outputs.hidden_states).permute(1,0,2,3)#[sample_i,i_layer,seqlen,dim]
                vis_tools.save_matrix(hidden_states_layers,epoch,args,mode="train",timestamp="new")
                # vis_tokenUni(outputs.hidden_states,batch["input_ids"],batch["labels"],tokenizer,picdir,ifpca,args,step)
            

        print("train_loss%.4f"%loss.item())
        end_time = datetime.datetime.now()
        time_list.append((end_time-start_time).seconds)
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            eval_loss = outputs.loss
            loss_test[str(loss_x)] = eval_loss.item()
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            ) 
        eval_metric = metric.compute()
        print("eval_loss%.4f"%eval_loss.item())
        logger.info(f"epoch {epoch}: {eval_metric}")
    print("****Cost Time Every Epoch****")
    print(sum(time_list)/len(time_list))
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    lossfile = os.path.join(args.output_dir,"{}_{}.txt".format(args.lnv,args.apply_exrank))
    np.savetxt(lossfile,np.array(list(loss_train.values())))
    # lossCurve(loss_train,loss_test,args)
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    predict_datasets = [predict_dataset]
    for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(args.output_dir, f"predict_results_{task}.txt")

        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {task} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:3.3f}\n")
                else:
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")
        
    

if __name__ == "__main__":
    main()
