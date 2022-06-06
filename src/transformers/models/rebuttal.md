1. We sincerely appreciate the reviewer's very constructive comments and suggestions. The following is our response in order of questions and comments raised. 

Q1: Add missing comparison with other methods.

 1-1) **Comparing with "adding exponential decay term in objective"** This method[1] is designed to solve the representation degeneration problem in language generation, i.e., machine translation task. They derive the singular value distribution of the output embedding matrix (decoder weight $W\in \mathbb{R}^{||V||\times d}$, where the $V$ is the vocabulary size). However, we alleviate the token uniformity issue to obtain better sentence-level representation, $x^{\ell}\in\mathbb{R}^{n_{\ell}\times m}$, $n_{\ell}$ is the feature dimension, $m$ is the number of tokens in the sentence (see in Chapter3). Although paper[1] is not directly applicable to our setup, we perform the experiment of adding the singular values $\{\sigma_{k} \}_{k=1}^K$ of  $x^{\ell}$ to the training objective as: 

$ \lambda_{e}\sum_{k=1}^{K}(\sigma_{k}-c_{1}e^{-c_{2}k^{\lambda}})$,  where the $\lambda_{e}$ is the hyper-parameter to adjust the auxiliary task weight. $c_1, c_2, \lambda$ are hyper-parameters in the desirable exponential prior term. Lacking for hyper-parameters in the GLUE tasks, we empirically set $c_{1},c_{2} =1, \lambda=2, \lambda_{e}=0.0001$ (The primary loss does not decrease until we decrease $\lambda_{e}$ to be 0.0001). From the GLUE results on BERT below, we doesn't see improvment from the sigular value decay term.
|           | cola  | sst2  | mrpc-acc | mrpc-f1 | qnli  | rte   |
|-----------|-------|-------|----------|---------|-------|-------|
| BERT      | 59.57 | 92.32 | 84.00    | 89.50   | 91.25 | 64.98 |
| BERT-decay| 59.37 | 92.43 | 83.25    | 87.92   | 89.21 | 64.98 |

It can be explained that 1) this method is essentially a multi-task model that is difficult to balance the two losses 2)it relies heavily on hyper-parameter in the prior decay term. Compared with our method, we only need to initialize $\alpha$ and it can be fine-tuned during training to fit the downstream tasks. For the STS dataset, our method can generate desirable results via directly transforming the output sentence representation without training on any corpus; which is not applicable to [1], so we do not perform comparision on this dataset.

1-2) **Combine our methods with SimCSE**. Thanks for your advice, it would be great if our model can further improve the SOTA models. Following the setup in SimCSE paper, we perform experiments in unsupervised and supervised manner on BERT. Combined with our method, the results are overall better than SimCSE, especially in the supervised setting. We attribute the superior superivsed results to using entailment pairs and their labels to better guide the fine-tuning the parameter $\alpha$ in our singular value transformation function. 
|        |       |       | unsup |       |       |       |        |
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|
|        | sts12 | sts13 | sts14 | sts15 | sts16 |  stsb | sick-r |
| SimCSE | 66.01 | 81.48 | 71.77 | 77.55 | 76.53 | 74.48 |  69.36 |
|   Our  | 63.25 | 78.67 | 70.41 | **79.37** | **77.69** | **75.81** |  **71.15** |
|        |       |       |  sup  |       |       |       |        |
| SimCSE | 77.37 | 78.12 | 77.81 | 84.65 | 81.10 | 82.26 |  78.73 |
|   Our  | 75.31 | **81.70** | **79.88** | **86.33** | **81.37** | **83.51** |  **79.04** |

Q2: Experiments on other pretrained models.

Thanks for valuable suggestions. We consider to add the Roberta results in a revised version. For the encoder-decoder structure, see in point 4. 

Q3: GLUE datasets

There are 3 kinds of tasks, including 9 datasets in GLUE benchmark. We have performed experiments on 6 of them (covering all the 3 tasks), except for QQP, MNLI and WNLI. The WNLI only consists of 634 training samples and is always excluded in the evaluation. We omit the other two in the current version due to its large size, but will add the results in the revised version.

Q4: Analysis of Cross-attention

Our paper explores the token unifomity issue in the information propagation in transformer encoder, mainly in the self-attention machanism (see in Chapter3). It would be very interesting to see it in the encoder-decoder structure and thus apply it to generation tasks. We think it would be a furture direction to improving the generation diversity via addressing the token uniformity, as 1) many existing work shows that anisotroy is related to the word frequence 2) In decodeing phase, sampling words from a more isotropic word embedding distribution would lead to a more diverse result.


2. We sincerely appreciate the reviewer's very constructive comments and suggestions. We aggree that an error analysis will improve the paper quality. For the application in generation tasks, our current version focuses on the encoder part, and it would be very interesting if we decode the words from a less anisotropic word embedding space. By doing so, the embeddings of the infrequent words would become more predominent and these words could gain larger probability to be generated. Therefore, we think it would be a futher direction between isotropy and generation diversity.

3. Thank you for your valuable feedback. We address your concerns:

Q1: Root cause of skewed distribution of singular values

We attribute the root cause of skewed distribution to the self-attention machanism, in which the softmax essentially calculate the weighted value vector of all input representations (See in [softmax bottleneck](https://arxiv.org/pdf/1711.03953.pdf)). The theory analysis in transformer-based can be found in [paper](https://arxiv.org/pdf/2103.03404.pdf). 

Q2: Token embeddings or singular values in Sec3.1

Sorry for the confusing expressions. We perform SVD on the output feature matrix $X^{\ell}$ and get its singular values $\{\lambda_i\}$. We seperate the token embeddings $X^{\ell}$ into two spanned singular value spaces $\chi_{[1,k]}^{\ell}$ and $\chi_{[k+1,m]}^{\ell}$. As we assume $X^{\ell}$ is full-rank, the size of $X^{\ell}$ and $\chi^{\ell}$ is the same, but they are in token embedding and singular value space, respectively. And there is no corresponding relation between $X^{\ell}_{i}$ and $\chi^{\ell}_{i}$.
Basically, $X^{\ell}_{i}$ is the linear combination of several $\chi^{\ell}_{j}$.



Q3: implementation of softdecay

We only insert the softdecay after the last output feature matrix. We performed experiments of inserting the softdecay after each output feature matrix, the perform improvment is insignificant but cost much more time. The softdecay function directly transform the singular value distribution of the output feature matrix, to make the singular value distribution more flatten. In some way, it is a post-processing method if the $\alpha$ is fixed. To gain better results, we make the $\alpha$ trainable in the downstream tasks to adaptively find the desirable flatness degree for different tasks. 







