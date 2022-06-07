from . import SentenceEvaluator, SimilarityFunction
import torch
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List
from ..readers import InputExample
from ..pooling_utils import soft_expand, whitening_torch_final
from .metrics import Evs, rbf_kernel, tokenSimi,struct_loss,nearbyloss,test
from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
import tensorboard as tb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

logger = logging.getLogger(__name__)

def evaluate_eigenvalue(embeddings1,embeddings2):
    """input embeddings:[n,dim]"""
    print("Statitics for all sentence_embeddings")
    input = torch.cat((embeddings1,embeddings2),axis=0)
    # print("Evs(Proportions of the top%d singular values):%.4f"%(1,Evs(input=input,k=1)))
    # print("Evs(Proportions of the top%d singular values):%.4f"%(3,Evs(input=input,k=3)))
    # print("rbf distance at t =2 %.4f"%rbf_kernel(input=input,unif_t=2))
    # print("Token Uniformity:%.4f"%tokenSimi(input))
    return Evs(input=input,k=1),Evs(input=input,k=3),rbf_kernel(input=input,unif_t=2),tokenSimi(input)


def tsne_map(embeddings1,embeddings2,dataset_name,stamps):
    model_type = "bert"
    input = torch.cat((embeddings1,embeddings2),axis=0)
    top1_evs = Evs(input=input,k=1)
    # top3_evs = Evs(input=input,k=3)
    rbf_dis = rbf_kernel(input=input,unif_t=2)
    tokenUni  = tokenSimi(input)
    print(stamps,top1_evs,rbf_dis,tokenUni)
    ifpca = True
    if ifpca:
        pca_50 = PCA(n_components=50)
        pca_result_50 = pca_50.fit_transform(input)
    tokens = pca_result_50
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tokens)
    #calculate the token_similarity. nopad==True:clip the [PAD]
    categories = input.shape[0]*["0"]
    categories[:embeddings1.shape[0]] = ["1"]*embeddings1.shape[0]
    d = {'tsne-2d-one': tsne_results[:,0], 'tsne-2d-two':tsne_results[:,1],"color_cate":categories}
    df_subset = pd.DataFrame(data=d)
    if stamps == "origin":
        colors = {'0':'tomato', '1':'black'}
    elif stamps == "soft_expand":
        colors = {'0':'blue', '1':'black'}
    elif stamps == "whitebert":
        colors = {'0':'green', '1':'black'}
    fig,ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_subset['tsne-2d-one'], df_subset['tsne-2d-two'],s=10, c=df_subset['color_cate'].map(colors))
    #add statistical results
    # plt.title("\n"+"Top1EVs:"+str("%.3f"%top1_evs)+" "+"RBF_dis:"+str("%.3f"%rbf_dis)+" "+"TokenUni:"+str("%.3f"%tokenUni),loc = "center")
    # picname = "tsne/"+model_type+"/"+stamps+"/"+dataset_name+"tsne_v2.png"
    # plt.savefig(picname)


class WhiteningEmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, trans_func: str= 'hanqi', main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, measure_data_num: int = -1, embed_dim: int = 768, summary_path: str = '', intra_diversity: bool = False):
    # def __init__(self, sentences1, sentences2, scores, batch_size, trans_func, main_similarity: SimilarityFunction = None, name = '', show_progress_bar = False, measure_data_num= -1, embed_dim = 768, summary_path = '', intra_diversity = False):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:
            List with the first sentence in a pair
        :param sentences2:
            List with the second sentence in a pair
        :param scores:
            Similarity score between sentences1[i] and sentences2[i]

        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.trans_func = trans_func

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name
        self.embed_dim = embed_dim
        self.intra_diversity = intra_diversity
        self.measure_data_num = measure_data_num
        self.summary_path = summary_path
        if self.summary_path:
            self.writer = SummaryWriter(summary_path)
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]
        # print("Initializing Evaluator with %s"%trans_func)

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        # logging.info("Evaluation the model on " + self.name + " dataset" + out_txt)

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=True)
        num_pairs = embeddings1.shape[0]

        embed = []
        meta_list = []
        num = 200
        if self.summary_path:
            meta_list.extend(["idx-{}<S1>{}".format(i, s1) for (i, s1) in zip(range(min(num_pairs, num)), self.sentences1[:num])])
            embed.append(embeddings1[:num, :])
            meta_list.extend(["idx-{}<S2>{}".format(i, s2) for (i, s2) in zip(range(min(num_pairs, num)), self.sentences2[:num])])
            embed.append(embeddings2[:num, :])
        
        # print("Using %s Transformation Method"%self.trans_func)
        # print("Before transformation:")
        # evaluate_eigenvalue(embeddings1,embeddings2)
        # origin_mask1 = struct_loss(embeddings1,3)
        # origin_mask2 = struct_loss(embeddings2,3)
        # tsne_map(embeddings1,embeddings2,self.name,stamps="origin")
        ori_embeddings = [embeddings1,embeddings2]
        if self.trans_func == "whitebert":
            embeddings = whitening_torch_final(torch.cat([embeddings1, embeddings2], dim=0))#embeddings1[n_samples,768]
            embeddings1 = embeddings[:num_pairs, :]
            embeddings2 = embeddings[num_pairs:, :]
        elif self.trans_func == "soft_decay":
            embeddings1 = soft_expand(embeddings1)
            embeddings2 = soft_expand(embeddings2)
        print("After transformation:")
        # evaluate_eigenvalue(embeddings1,embeddings2)
        # tsne_map(embeddings1, embeddings2,self.name,stamps=self.trans_func)
        # ori_error1,t_error1 = nearbyloss(ori_embeddings[0],embeddings1)
        # ori_error2,t_error2 = nearbyloss(ori_embeddings[1],embeddings2)
        # print(ori_error1,t_error1)
        # print(ori_error2,t_error2)
        # tranformed_mask1 = struct_loss(embeddings1,3)
        # transformed_mask2 = struct_loss(embeddings2,3)
        # """element-wise multiplication->summation->larger means hits more"""
        # intra_simi1 = sum(torch.sum(origin_mask1*tranformed_mask1,dim=-1))
        # intra_simi2 = sum(torch.sum(origin_mask2*transformed_mask2,dim=-1))
        # inter_simi1 = sum(torch.sum(origin_mask1*transformed_mask2,dim=-1))
        # inter_simi2 = sum(torch.sum(origin_mask2*tranformed_mask1,dim=-1))
        # print(intra_simi1/origin_mask1.shape[0],intra_simi2/origin_mask1.shape[0],inter_simi1/origin_mask1.shape[0],inter_simi2/origin_mask1.shape[0])
        if self.summary_path:
            meta_list.extend(["white-idx-{}<WS1>{}".format(i, s1) for (i, s1) in zip(range(min(num_pairs, num)), self.sentences1[:num])])
            embed.append(embeddings1[:num, :])
            meta_list.extend(["white-idx-{}<WS2>{}".format(i, s2) for (i, s2) in zip(range(min(num_pairs, num)), self.sentences2[:num])])
            embed.append(embeddings2[:num, :])
            embed = torch.cat(embed, dim=0)
            self.writer.add_embedding(embed, metadata=meta_list, tag="all{}".format(num*4))
        embeddings1 = embeddings1[:self.measure_data_num, :self.embed_dim]
        embeddings2 = embeddings2[:self.measure_data_num, :self.embed_dim]
        labels = self.scores[:self.measure_data_num]

        if self.intra_diversity:
            intra_div = self.compute_intra_diversity(embeddings1, embeddings2)
            logging.info("IntraDiversity on "+self.name+out_txt+": {:.4f}".format(intra_div))
            return intra_div

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)


        logging.info("Eval on "+self.name+out_txt+"Cosine :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        # logging.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_manhattan, eval_spearman_manhattan))
        # logging.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_euclidean, eval_spearman_euclidean))
        # logging.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_dot, eval_spearman_dot))
        # logging.info("Eval on "+self.name+out_txt+"Cosine3 :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        #     eval_pearson_cosine3, eval_spearman_cosine3))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                       writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])


        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")

    def compute_intra_similarity(self, model):
        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, output_value="intra_similarity",
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, output_value="intra_similarity",
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        return np.mean(embeddings1), np.mean(embeddings2)

    def compute_intra_diversity(self, embeddings1, embeddings2):
        embedding = np.concatenate([embeddings1, embeddings2], axis=0)
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        diversity = np.mean(embedding.dot(embedding.T))
        return diversity
