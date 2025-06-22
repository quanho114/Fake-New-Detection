# 📰 Fake News Detection using BERT & XLNet

This project focuses on detecting fake news using state-of-the-art transformer-based language models — BERT and XLNet. Fake news has become a serious challenge in the digital age, and automatic detection systems can play a crucial role in mitigating misinformation.

We fine-tune two powerful pretrained models, BERT (Bidirectional Encoder Representations from Transformers) and XLNet (Generalized Autoregressive Pretraining for Language Understanding), to perform binary classification on news articles — classifying them as either **FAKE** or **REAL**. The project involves comprehensive data preprocessing, tokenization, model training, and evaluation using real-world news data.

By leveraging transfer learning and modern NLP architectures, this project demonstrates how transformer models can achieve high accuracy and robust performance in a practical natural language understanding task.


---

## 📌 Overview

- **Goal**: Classify news articles as **FAKE** or **REAL**
- **Models used**:
  - BERT (Bidirectional Encoder Representations from Transformers)
  - XLNet (Generalized Autoregressive Pretraining for Language Understanding)
- **Dataset**: https://drive.google.com/drive/folders/1mrX3vPKhEzxG96OCPpCeh9F8m_QKCM4z
- **Frameworks**: PyTorch, HuggingFace Transformers, Scikit-learn

---

## 📌 Project Workflow

1. **Final Project Objective**  
   Build a binary classifier to detect fake news articles using pretrained NLP models (BERT and XLNet).

2. **Load Data**  
   - Load and inspect the Fake & Real News dataset from Kaggle.

3. **Data Processing**  
   - Clean text data (remove punctuation, lowercasing, stopwords, etc.).
   - Label encoding (FAKE = 0, REAL = 1).

4. **Exploratory Data Analysis (EDA)**  
   - 📊 **3.1 Class Distribution**: Visualize the balance between fake and real samples.  
   - ✏️ **3.2 Text Length Analysis**: Check distribution of article lengths.  
   - ☁️ **3.3 Word Cloud Analysis**: Display most frequent terms in FAKE vs. REAL news.  
   - 📚 **3.4 Most Common Words**: Analyze top unigrams and bigrams per class.

5. **Model Building**  
   - Use HuggingFace Transformers for BERT and XLNet.
   - Tokenization with appropriate model tokenizer.

6. **Hyperparameter Setup**  
   - Define learning rate, batch size, epochs, max sequence length, etc.

7. **Model Training**  
   - Fine-tune BERT and XLNet using PyTorch.
   - Apply early stopping and validation monitoring.

8. **Performance Evaluation**  
   - Compute accuracy, precision, recall, F1-score.
   - Visualize confusion matrix for both models.

9. **Inference**  
   - Use trained models to predict fake/real labels on unseen news samples.

---

## 📊 Features Summary

- ✅ End-to-end NLP pipeline for binary classification
- 🧹 Robust text preprocessing and cleaning
- 📊 Visual EDA: class distribution, word clouds, length histograms
- 🤖 Fine-tuning of transformer models (BERT & XLNet)
- ⚙️ Custom hyperparameter configuration
- 📈 Model evaluation with multiple metrics
- 🔍 Real sample inference using trained models
- 📊 Confusion matrix visualizations for comparison

---

## 📈 Evaluation Results

### 🔹 BERT Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fake  | 0.96      | 0.96   | 0.96     | 5105    |
| True  | 0.96      | 0.96   | 0.96     | 5179    |
| **Overall Accuracy** |       |        | **0.96** | 10284   |

### 🔹 XLNet Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fake  | 0.94      | 0.98   | 0.96     | 5105    |
| True  | 0.97      | 0.94   | 0.96     | 5179    |
| **Overall Accuracy** |       |        | **0.96** | 10284   |

### 🔍 Comparison Summary

- Both models achieved **96% overall accuracy** on the test set.
- **XLNet** had slightly higher precision on the **True** class but slightly lower recall.
- **BERT** showed more balanced performance across both classes.
- Final F1-scores are equal (0.96), indicating both models are strong candidates for deployment.

---

## 🚀 Quick Start

### ✅ Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1VJzW7t9ZLBDxsdBPwlqDja7ubcJxKrLa/view?usp=sharing)

### 🔧 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/quanho114/Fake-New-Detection.git
cd Fake-New-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook fake_news_bert_xlnet.ipynb
