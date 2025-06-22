# 📰 Fake News Detection using BERT & XLNet

This project applies cutting-edge transformer-based models (BERT and XLNet) to the task of **fake news detection** in English-language news articles. It demonstrates how powerful pretrained language models can be fine-tuned for binary classification tasks in Natural Language Processing (NLP).

---

## 📌 Overview

- **Goal**: Classify news articles as **FAKE** or **REAL**
- **Models used**:
  - BERT (Bidirectional Encoder Representations from Transformers)
  - XLNet (Generalized Autoregressive Pretraining for Language Understanding)
- **Dataset**: [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Frameworks**: PyTorch, HuggingFace Transformers, Scikit-learn

---

## 📊 Features

- Data cleaning and preprocessing
- Tokenization using HuggingFace Transformers
- Fine-tuning BERT and XLNet on fake news classification
- Binary classification: FAKE vs. REAL
- Evaluation with multiple metrics (Accuracy, Precision, Recall, F1-score)
- Confusion matrix visualization
- Performance comparison between BERT and XLNet

---

## 📈 Evaluation Results

| Model  | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| BERT   | 95.2%    | 95.4%     | 94.9%  | 95.1%    |
| XLNet  | 96.1%    | 96.0%     | 96.2%  | 96.0%    |

📌 *Note: Results may vary depending on random seed, dataset split, and training configuration.*

---
## 🤔 Key Challenges

- Preprocessing long text sequences and truncation
- Handling class imbalance in fake/real distribution
- Managing GPU memory with large transformer models

## 💡 Insights

- XLNet slightly outperformed BERT, possibly due to its permutation-based training.
- Fine-tuning only the last few layers of BERT gave almost similar results while saving training time.

---

## 🔮 Future Work

- Apply RoBERTa and DeBERTa for performance comparison
- Integrate attention visualization for explainability
- Deploy model as a web API using FastAPI or Flask

---



## 🚀 Quick Start

### ✅ Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo-name/blob/main/fake_news_bert_xlnet.ipynb)

### 🔧 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook fake_news_bert_xlnet.ipynb
