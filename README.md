# Fake vs. Real News Classification Using Random Forest

DS 4002  
Group Name: DLC  
Group Leader: Lauren Medica  
Group Members: Lauren Medica, Dev Patel, Caroline Lingle  

---

## Project Overview

This project investigates whether fake news is easier to detect using full article text or only the article title. Using the Kaggle "Fake-and-Real-News-Dataset", we build and compare two Random Forest classification models.

Our hypothesis is that it is harder to detect fake news using only titles than using full article text. Model performance is evaluated primarily using the F1 score.

---

## Repository Structure

This repository follows a structured format to ensure reproducibility.

Project Root  
│  
├── README.md  
├── LICENSE.md  
│  
├── SCRIPTS/  
│   ├── 01_load_data.py  
│   ├── 02_preprocessing.py  
│   ├── 03_full_text_model.py  
│   ├── 04_title_model.py  
│   ├── 05_model_comparison.py  
│   ├── 06_make_data_appendix.py  
│  
├── DATA/  
│   ├── raw/  
│   │   ├── fake.csv  
│   │   ├── real.csv  
│   ├── processed/  
│   │   ├── cleaned_data.csv  
│   ├── Data_Appendix.pdf  
│  
├── OUTPUT/  
│   ├── confusion_matrix_full.png  
│   ├── confusion_matrix_title.png  
│   ├── metrics_comparison.csv  
│   ├── feature_importance_full.png  
│   ├── feature_importance_title.png  

---

## Section 1: Software and Platform

This project was developed using:

- Python 3.10+
- Google Colab or VS Code
- Windows or Mac OS

Required Python packages:

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- nltk  

To install required packages:

```
pip install pandas numpy scikit-learn matplotlib seaborn nltk
```

---

## Section 2: Data Source

Dataset used:

C. Bisaillon, “Fake-and-Real-News-Dataset,” Kaggle, 2024.  
Available: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  

Unit of observation: One news article.

Each observation includes:

- title  
- text  
- subject  
- date  
- label (fake or real)

---

## Section 3: Instructions to Reproduce Results

Follow these steps to reproduce our results:

### Step 1: Download Dataset

Download the Kaggle dataset and place the following files into:

DATA/raw/

- fake.csv  
- real.csv  

---

### Step 2: Run Scripts in Order

From the project root directory, execute the scripts in this exact order:

```
python SCRIPTS/01_load_data.py
python SCRIPTS/02_preprocessing.py
python SCRIPTS/03_full_text_model.py
python SCRIPTS/04_title_model.py
python SCRIPTS/05_model_comparison.py
python SCRIPTS/06_make_data_appendix.py
```

Each script must be run sequentially.

---

### Step 3: Generated Output

After running all scripts, results will be saved in:

OUTPUT/

Including:

- Confusion matrices  
- Model performance metrics  
- Feature importance plots  
- Model comparison table  

The Data Appendix PDF will be generated in:

DATA/Data_Appendix.pdf

---

## Modeling Approach

Two Random Forest classifiers were trained:

1. Full-text model  
2. Title-only model  

Both models:

- Use TF-IDF vectorization  
- Use an 80/20 train-test split  
- Use identical preprocessing steps  
- Are evaluated using F1 score  

Our quantifiable goal was to achieve an F1 score ≥ 0.90 for the full-text model.

---

## Evaluation Metrics

We evaluate performance using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

F1 score is emphasized because it balances precision and recall and is appropriate for classification tasks involving potential class imbalance.

---

