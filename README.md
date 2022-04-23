# tokenUni
Isotropy_TokenUni

We sincerely appreciate the reviewer's very constructive comments and suggestions. The following is our response in order of questions and comments raised. <br>
Q1: Add missing comparison with other methods.<br>
<br>
Thank you for the valuable suggestion. The method of "adding exponential decay term in objective" is designed to solve the representation degeneration problem in language generation, i.e., machine translation task in the original paper. Therefore, they derive the singular value distribution of the output embedding matrix (decoder weight $W\in \mathbf{R}^{||V||\times d}$)

<br>
Q2: Add experiments on other pre-trained language model such as RoBERTa or T5.<br>
Q3: Report full results on GLUE.<br>
Q4: Extend analysis to transformer cross-attention.<br>

