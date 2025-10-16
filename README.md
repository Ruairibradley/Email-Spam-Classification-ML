# Spam-Email-Detector
Email Spam Classification system using NLP and machine learning techniques.

# Email Spam Classification using NLP and Machine Learning

This project implements a spam email detection system using Natural Language Processing (NLP) and supervised machine learning.  
The model classifies emails as either *spam* or *ham (not spam)* by analysing their textual content through tokenisation, vectorisation, and predictive modelling.

Developed as part of the *Applied Machine Learning* module at the University of Sussex (2025).

---
## Full Report

See section of the report relevant to spam detection.
A detailed technical report discussing the experimental design, model selection, and evaluation is available here:

[Download Full Report (PDF)](./Report.pdf)

The report includes:
- Background on spam detection and NLP techniques  
- Implementation details for feature extraction and model training  
- Quantitative performance analysis and visual results  
- Discussion of accuracy, precision, recall, and F1-score findings


## Overview

The project demonstrates how NLP techniques and traditional ML algorithms can be used to detect spam messages effectively.  
A dataset of labeled emails was preprocessed, transformed into numerical representations using TF-IDF vectorisation, and then used to train multiple classifiers.

After experimentation with several algorithms, the best-performing model achieved almost perfect accuracy, indicating highly effective text feature extraction and generalisation.

---

## Implementation Details

Language: Python  
Libraries: `pandas`, `scikit-learn`, `NumPy`, `matplotlib`, `nltk`

Key Steps:
1. Data Preprocessing:
   - Cleaning text (removing punctuation, stopwords, and HTML tags).
   - Lemmatization and lowercasing for uniformity.
2. Feature Engineering:
   - Tokenisation and TF-IDF vectorisation to convert text into numerical features.
3. Model Training:
   - Trained and compared multiple classifiers including Logistic Regression, Multinomial Naïve Bayes, SVM, and Random Forest.
4. Model Evaluation:
   - Evaluated using precision, recall, F1-score, and accuracy metrics.
   - Performed hyperparameter tuning to maximise performance.
5. Prediction:
   - Classified unseen messages with real-time predictions and confidence scores.

---

## Model Performance

The final optimised model achieved outstanding results:

| Metric | Score |
|:-------|:------:|
| **Accuracy** | **1.00** |
| **Precision (Class 0)** | **1.00** |
| **Precision (Class 1)** | **0.99** |
| **Recall (All Classes)** | **1.00** |
| **F1-Score (Weighted Avg)** | **1.00** |

*Result: The model correctly classified nearly every email in the test set, achieving an F1-score of 1.00, indicating perfect spam/ham separation.*

---

## How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Ruairibradley/Email-Spam-Classification.git
   cd Email-Spam-Classification

2. Open the file in your IDE:
  AML_Code_Task2.ipynb
3. Follow the notebook instructions and run code sequentially.
4. View Results.
5. Can also upload file to Google Colab instantly for no setup required. 


