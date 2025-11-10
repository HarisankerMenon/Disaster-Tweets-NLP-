# 🧠 Disaster Tweets Classification (NLP Project)

This project focuses on building a Natural Language Processing (NLP) model that can classify tweets as **disaster-related** or **not disaster-related** using real-world Twitter data. It’s inspired by the Kaggle competition *“Real or Not? NLP with Disaster Tweets”*.

---

## 🎯 Project Objective
The goal is to develop a machine learning model that accurately predicts whether a tweet describes a real disaster event based on its text content.

---

## 📂 Dataset
- **Source:** Kaggle (https://www.kaggle.com/competitions/nlp-getting-started)
- **Records:** 10,000+ labeled tweets
- **Columns:** `text`, `location`, `keyword`, `target`

---

## ⚙️ Tools & Technologies
- Python 3  
- Pandas, NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib, Seaborn  

---

## 🧩 Project Workflow
1. **Data Cleaning & Preprocessing**
   - Removed URLs, hashtags, mentions, and punctuation  
   - Tokenized and lemmatized tweets  
   - Removed stopwords  

2. **Exploratory Data Analysis (EDA)**
   - Word frequency and length distribution plots  
   - Correlation between tweet text and labels  

3. **Feature Engineering**
   - TF-IDF vectorization  
   - Bag-of-Words representation  

4. **Model Training**
   - Logistic Regression, Naive Bayes, Random Forest  
   - Hyperparameter tuning  

5. **Evaluation**
   - Accuracy, F1-score, Confusion Matrix visualization  

---

## 📈 Results
- **Best Model:** Logistic Regression  
- **Accuracy:** 0.83  
- **F1 Score:** 0.82  

---

## 🧠 Insights
- Tweets with strong emotional or location-based keywords were more predictive of real disasters.  
- Text normalization and feature engineering significantly improved performance.  

---

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
jupyter notebook Disaster_Tweets.ipynb
