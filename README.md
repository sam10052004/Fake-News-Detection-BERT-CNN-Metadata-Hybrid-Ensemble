# Fake News Detection using NLP (BERT + CNN + Metadata) 📰

This project focuses on identifying **Fake News** using a hybrid deep learning approach that combines:

  * **BERT** for contextual language understanding
  * **CNN** for linguistic pattern extraction
  * **Metadata**-based source credibility scoring
  * **Ensemble fusion** for final prediction

This hybrid strategy achieves highly reliable results and minimizes misclassification.

-----

## Dataset 📚

Dataset Used: **Fake and Real News Dataset (Kaggle)**
Link: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Files included:

  * `Fake.csv`
  * `True.csv`

-----

## Features ✨

| Module | Description |
|--------|-------------|
| Data Preprocessing | Text cleaning, normalization, stopword removal |
| Metadata Credibility | Assigns reliability score based on news publisher |
| BERT Fine-Tuning | Learns contextual embeddings from text |
| CNN Model | Learns phrase-level and semantic patterns |
| Ensemble Model | Combines BERT + CNN predictions |
| Evaluation | Generates accuracy & classification report |

-----

## Model Performance 🚀

| Model | Accuracy |
|-------|---------|
| BERT + Metadata | 99.97% |
| CNN Model | 99.01% |
| Ensemble (Final) | **99.97% ✅** |

**Classification Report:**

```
precision recall f1-score support
1.00 1.00 1.00 (7821 samples)

Overall Accuracy: 1.00
```

-----

## Project Architecture 📐

```
Raw News Text
│
Text Preprocessing
│
┌───────────────┬─────────────────┐
│               │                 │
BERT Model Metadata Score CNN Model
│               │                 │
└───────────────┴─────────────────┘
Ensemble Fusion
│
Fake / Real
```

-----

## How to Run (Google Colab) 💻

1.  Open the notebook:
    `Fake_News_Detection_Colab.ipynb`

2.  Set runtime to GPU:
    `Runtime` → `Change runtime type` → `GPU`

3.  Upload `archive.zip` dataset to Colab

4.  Run all cells in order

-----

## Requirements 📦

  * transformers
  * torch
  * tensorflow
  * keras
  * pandas
  * numpy
  * scikit-learn
  * matplotlib

Install using:

```bash
pip install transformers torch tensorflow pandas numpy scikit-learn
```

-----

## Folder Structure 🗂️

```
├── Fake_News_Detection_Colab.ipynb
├── README.md
├── models/
│ ├── bert_metadata_model.pt
│ └── cnn_fake_news_model.h5
└── dataset/ (optional)
```

-----

## Conclusion

This project demonstrates an **accurate and scalable approach to Fake News Detection** by integrating NLP-based contextual modelling, pattern extraction using CNNs, and credibility-based metadata scoring. The ensemble model achieves near perfect classification performance and is suitable for real-world media validation applications.
