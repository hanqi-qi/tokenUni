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

logger = logging.getLogger(__name__)

class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, embed_dim: int = 768, intra_similarity: bool = False, intra_diversity: bool = False):
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

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name
        self.embed_dim = embed_dim
        self.intra_similarity = intra_similarity
        self.intra_diversity = intra_diversity

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

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

        embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores
        if self.intra_similarity:
            intra_sim1, intra_sim2 = self.compute_intra_similarity(model)
            intra_sim = (intra_sim1 + intra_sim2) / 2
            logging.info("IntraSimilarity on "+self.name+out_txt+": {:.4f}".format(intra_sim))
            return intra_sim

        if self.intra_diversity:
            intra_div = self.compute_intra_diversity(embeddings1, embeddings2)
            logging.info("IntraDiversity on "+self.name+out_txt+": {:.4f}".format(intra_div))
            return intra_div

        embeddings1 = embeddings1[:, :self.embed_dim]
        embeddings2 = embeddings2[:, :self.embed_dim]

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
