# Addressing Token Uniformity in Transformers via Singular Value Transformation
UAI2022: Addressing Token Uniformity in Transformers via Singular Value Transformation 

[Hanqi Yan](https://warwick.ac.uk/fac/sci/dcs/people/u2048587/), [Lin Gui](https://warwick.ac.uk/fac/sci/dcs/people/lin_gui/), [Wenjie Li](https://www4.comp.polyu.edu.hk/~cswjli/), [Yulan He](https://warwick.ac.uk/fac/sci/dcs/people/yulan_he/).

In this work, we characterise the token uniformity problem commonly observed in the output of transformer-based architectures by the degree of skewness in singular value distributions and propose a singular value transformation function (SoftDecay) to address the problem


## Requirements

Our project is based on Huggingface Transformers, so basically you can refer the requirements from [their repository](https://github.com/huggingface/transformers). 

## Code Structure

This repo contains three primary parts to reproduce our numeric results and visualize singular value distribution on different tasks, i.e., main functions using to call different pretrained language models (vanilla and +SoftDecay), in [path_extract repository](path_extract), incorporat the knowledge paths to capturing the causal relations between the document clauses in [main.py](main.py), generate adversarial samples and evaluate existing ECE models on these sampels in [ad_paedgl.py](ad_paedgl.py). 

### Path Extraction
We first extract knowledeg paths which contain less than two intermediate entities from ConceptNet. [KagNet](https://github.com/INK-USC/KagNet) contains the driver code to extract all the knowledge paths between the given head entity and the tail entity. Our code provides how to identity the keywords as the head/tail entity, and the path filter mechanism in [filter_path.py](path_extract/filter_path.py).

### Knowledge-aware graph model
This part use the extracted paths to identity the cause clauses in a document.
```
python main.py
```
To see the model performances with absolute position rather than the relative position, modify the input data in [model_funcs.py](utils/model_funcs.py).
### Adversarial Attacks
This part genetrate the adversarial samples, which swap two clauses to disturb the original relative position information. Then we observe performances drops in exsiting ECE models. Use PAEDGL model as an example.
```
python ad_paedgl.py
```

## Citation

If you find our work useful, please cite as:

```
@misc{yan2021position,
      title={Position Bias Mitigation: A Knowledge-Aware Graph Model for Emotion Cause Extraction}, 
      author={Hanqi Yan and Lin Gui and Gabriele Pergola and Yulan He},
      year={2021},
      eprint={2106.03518},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
