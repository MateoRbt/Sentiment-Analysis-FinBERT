# Sentiment Analysis with FinBERT, MLP & BiLSTM
## OVERVIEW 
This project explores multi-class sentiment classification (positive, negative, neutral) on financial text using various deep learning approaches.

## Models Compared
1 ) FinBERT (fine-tuned) – with Focal Loss & class weights

2) MLP + FinBERT embeddings – with SMOTE oversampling

3) BiLSTM – trained on FinBERT embeddings with Focal Loss

## Evaluation Summary
| Model      | Accuracy | F1 Score | Precision | Recall | Time      |
| ---------- | -------- | -------- | --------- | ------ | --------- |
| FinBERT    | 82.28%   | 0.8317   | 0.8691    | 0.8228 | \~5 min   |
| MLP        | 81.51%   | 0.7871   | 0.7762    | 0.8343 | \~1.4 sec |
| **BiLSTM** | 80.14%   | 0.8120   | 0.8501    | 0.8014 | \~10 sec  |

All models trained with stratified splits (80/20) and early stopping.

## Key Techniques

Text preprocessing (NLTK, regex, class weights)

FinBERT for embeddings & classification

SMOTE for class imbalance (MLP only)

Focal Loss for hard samples (FinBERT & BiLSTM)

Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix

## Resources
Dataset: https://www.kaggle.com/code/mateorbt/sentiment-analysis-deep-learning
Binary classification: https://github.com/MateoRbt/Sentiment-Analysis
Focal Loss: https://github.com/itakurah/Focal-loss-PyTorch
Repo for code : https://github.com/MateoRbt/Sentiment-Analysis-FinBERT
