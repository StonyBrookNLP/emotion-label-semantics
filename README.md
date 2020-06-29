
# Modeling-Label-Semantics-for-Predicting-Emotional-Reactions

Code for the paper - Modeling Label Semantics for Predicting Emotional Reactions - https://arxiv.org/pdf/2006.05489.pdf. This work explores the use different ways of utilizing label semantics techniques for improving multi-label emotion prediction in short commonsense stories:

![GitHub Logo](images/common-sense-stories-new.png)
*Multi-label emotion reactions in ROC Stories (Rashkin et al.)*

## Running experiments
### Training a model

This section covers training a classification model for the Semi-supervision model and the baseline models (refer to Table 1 in the paper). The model names and corresponding main file names are mentioned below:
1. BERT - main_bert_base.py
2. LEAM w/ Glove - main_leam_glove_label.py
3. LEAM w/ BERT Features - main_leam_bert_label.py
4. BERT + Labels as Input - main_bert_label.py
5. Learned Correlations - main_bert_learned_correlation.py
6. Semi-supervision - main_bert_soft_semi_supervision.py

Run the following command for training each of above listed models with the correct main file:
```
python main_bert_soft_semi_supervision.py --do_train --save_best --do_lower_case --monitor valid_loss --mode min --epochs 20 --train_batch_size 8
```
### Code citations
This work wouldn't be possible without the code shared in the following repositories, which were used heavily in this work
* The code for multi-label classification with a BERT pretrained model was taken from:
https://github.com/lonePatient/Bert-Multi-Label-Text-Classification. The initial framework provided here was used for developing all our bert-based models
* The code for the baselines in [Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/), was taken from:
https://github.com/atcbosselut/scs-baselines
* The following repository - https://github.com/guoyinwang/LEAM helped in understanding the LEAM architecture. Our work has re-implemented this framework in Pytorch with slight modifications (refer to the paper)
* The code for an additional baseline - [Ranking and Selecting Multi-Hop Knowledge Paths to Better PredictHuman Needs](https://www.aclweb.org/anthology/N19-1368.pdf) was referenced from - https://github.com/debjitpaul/Multi-Hop-Knowledge-Paths-Human-Needs . We change this code to Pytorch and modified the initial implementation to account for the Plutchik emotion prediction
