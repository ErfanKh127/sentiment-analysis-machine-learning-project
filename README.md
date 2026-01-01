# 🎬 Sentiment Analysis of Movie Reviews
A Machine Learning Project Using NLTK & Scikit-Learn

## 📌 Overview
This project implements a **sentiment analysis classifier** that predicts whether a movie review is **positive** or **negative**.  
It uses the **NLTK movie_reviews corpus**, **TF-IDF vectorization**, and a **Naive Bayes** model to classify review text.

---

## 🚀 Features
- Preprocesses raw text (cleaning, stopword removal)
- Converts text to numerical features using **TF-IDF**
- Trains a **Multinomial Naive Bayes** sentiment classifier
- Evaluates model performance using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 🧰 Technologies Used
- **Python**
- **NLTK**
- **Scikit-Learn**
- **Pandas**
- **NumPy**

---

## 📁 Dataset
This project uses the **movie_reviews** dataset from NLTK, which contains:

- 2,000 movie reviews  
- Each labeled as **pos** (positive) or **neg** (negative)

---

## 📚 Project Workflow

### 1. Import Dependencies
Loads necessary libraries for text processing and machine learning.

### 2. Load the Dataset  
Retrieves movie reviews and associated sentiment labels from NLTK.

### 3. Text Preprocessing  
- Lowercasing  
- Removing punctuation  
- Removing stopwords  
- Tokenization  
- Joining tokens into clean text  

### 4. Feature Extraction  
Text is converted into numerical vectors using **TfidfVectorizer**.

### 5. Train/Test Split  
The dataset is divided into:
- **80% training data**
- **20% testing data**

### 6. Model Training  
A **Multinomial Naive Bayes** classifier is trained on the processed TF-IDF features.

### 7. Model Evaluation  
Performance metrics printed:
- Accuracy score  
- Confusion matrix  
- Classification report  

---

## 📊 Results
Typical accuracy for this model ranges from **80% to 86%**, depending on data split and preprocessing.

---

## 📦 How to Run This Project

### 1. Install dependencies
```bash
pip install nltk scikit-learn pandas
