# 🚨 Fake Job Detection System

This project is an NLP-based Fake Job Detection System that classifies job postings as **Real** or **Fake** using Machine Learning.

## 🔍 Problem Statement
Fake job postings are increasing on online platforms and often scam job seekers.  
This system helps identify fraudulent job posts automatically.

## 🧠 Technologies Used
- Python
- Natural Language Processing (NLP)
- TF-IDF
- Logistic Regression
- BERT (optional)
- Streamlit

## ⚙️ How It Works
1. User enters a job description
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical features
4. ML model predicts Real/Fake
5. Key indicators are shown for explainability

## 🖥️ Application Features
- Real-time prediction
- Confidence score
- Explainable keyword indicators
- Clean UI with animations
- Input validation
- Admin dashboard (optional)

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py


## 📊 Dataset Information

The dataset used in this project is too large to upload to GitHub.

You can download it from:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

After downloading, place the file in the project root directory and rename it as:

fake_job_postings.csv

