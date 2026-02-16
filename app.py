import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import os
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import time

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🚨",
    layout="wide"
)

# ======================================================
# CUSTOM CSS (SPACING + ANIMATIONS)
# ======================================================
st.markdown("""
<style>
.section {
    margin-top: 40px;
    margin-bottom: 40px;
}
.result-card {
    background: linear-gradient(145deg, #0f0f14, #1a1a22);
    padding: 28px;
    border-radius: 16px;
    margin-top: 30px;
    margin-bottom: 30px;
    box-shadow: 0 0 25px rgba(0,0,0,0.6);
    border-left: 6px solid;
    animation: slideFade 0.8s ease-out;
}
.fake {
    border-color: #ff4b4b;
    animation: glowRed 1.5s infinite alternate;
}
.real {
    border-color: #2ecc71;
    animation: glowGreen 1.5s infinite alternate;
}
.result-title {
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 18px;
}
.result-text {
    font-size: 18px;
    margin: 10px 0;
}
.highlight {
    background-color: #ff4b4b;
    color: white;
    padding: 4px 8px;
    border-radius: 6px;
    margin: 0 2px;
}
@keyframes slideFade {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes glowRed {
    from { box-shadow: 0 0 10px rgba(255,75,75,0.3); }
    to   { box-shadow: 0 0 28px rgba(255,75,75,0.8); }
}
@keyframes glowGreen {
    from { box-shadow: 0 0 10px rgba(46,204,113,0.3); }
    to   { box-shadow: 0 0 28px rgba(46,204,113,0.8); }
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "user"
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prediction_label" not in st.session_state:
    st.session_state.prediction_label = None
if "prediction_prob" not in st.session_state:
    st.session_state.prediction_prob = 0.0
if "bert_score" not in st.session_state:
    st.session_state.bert_score = 0.0
if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

# ======================================================
# ADMIN LOGIN
# ======================================================
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ======================================================
# NLTK SETUP
# ======================================================
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ======================================================
# LOAD MODEL
# ======================================================
model = pickle.load(open("fake_job_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ======================================================
# FUNCTIONS
# ======================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

def is_valid_job_description(text):
    return len(text.split()) >= 30, "Minimum 30 words required."

def bert_predict(text):
    from transformers import pipeline
    bert = pipeline("text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english")
    result = bert(text[:512])[0]
    return (1, result["score"]) if result["label"] == "NEGATIVE" else (0, result["score"])

def hybrid_predict(text, use_bert):
    vec = tfidf.transform([clean_text(text)])
    ml_pred = model.predict(vec)[0]
    ml_prob = model.predict_proba(vec)[0][1]
    bert_score = 0
    final_pred = ml_pred
    if use_bert:
        bert_pred, bert_score = bert_predict(text)
        final_pred = 1 if (ml_pred == 1 or bert_pred == 1) else 0
    return final_pred, ml_prob, bert_score

def save_feedback(job_text, prediction, feedback, comment):
    df = pd.DataFrame([{
        "timestamp": datetime.now(),
        "prediction": prediction,
        "user_feedback": feedback,
        "comment": comment,
        "job_text": job_text[:500]
    }])
    df.to_csv("feedback.csv", mode="a",
              header=not os.path.exists("feedback.csv"),
              index=False)

def clear_feedback_data():
    if os.path.exists("feedback.csv"):
        os.remove("feedback.csv")
        return True
    return False

# ======================================================
# SUSPICIOUS KEYWORDS
# ======================================================
SUSPICIOUS_WORDS = [
    "urgent", "work from home", "earn", "no experience",
    "immediate joining", "bank account", "limited vacancies",
    "daily payment", "guaranteed","experience", "requirements", "responsibilities",
    "skills", "qualification", "salary", "location",
    "sql", "python", "azure", "etl", "adf", "shift"
]

def highlight_keywords(text):
    for word in SUSPICIOUS_WORDS:
        text = re.sub(word,
                      f"<span class='highlight'>{word}</span>",
                      text, flags=re.IGNORECASE)
    return text

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("📌 Navigation")
if st.sidebar.button("🏠 User Prediction"):
    st.session_state.page = "user"
if st.sidebar.button("📊 Admin Dashboard"):
    st.session_state.page = "admin"

# ======================================================
# HEADER
# ======================================================
st.markdown("<h1 style='text-align:center;'>🚨 Fake Job Detection System</h1>",
            unsafe_allow_html=True)

# ======================================================
# USER PAGE
# ======================================================
if st.session_state.page == "user":

    use_bert = st.checkbox("🧠 Enable BERT (slower but accurate)")
    job_text = st.text_area("📄 Job Description", height=260)

    if st.button("🔍 Predict"):
        valid, msg = is_valid_job_description(job_text)
        if not valid:
            st.warning(msg)
        else:
            with st.spinner("🔄 Analyzing job description..."):
                time.sleep(1.2)
                final_pred, ml_prob, bert_score = hybrid_predict(job_text, use_bert)

            st.session_state.predicted = True
            st.session_state.prediction_label = "Fake" if final_pred == 1 else "Real"
            st.session_state.prediction_prob = ml_prob
            st.session_state.bert_score = bert_score

    if st.session_state.predicted:

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)

        label = st.session_state.prediction_label
        ml_prob = round(st.session_state.prediction_prob * 100, 2)
        bert_score = round(st.session_state.bert_score * 100, 2)

        st.markdown(f"""
        <div class="result-card {'fake' if label=='Fake' else 'real'}">
            <div class="result-title">
                {'❌ FAKE JOB' if label=='Fake' else '✅ REAL JOB'}
            </div>
            <div class="result-text">ML Probability: <b>{ml_prob}%</b></div>
            <div class="result-text">BERT Score: <b>{bert_score}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("📊 Confidence Level")
        st.progress(int(ml_prob))

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("🧠 Suspicious Keywords Detected")
        st.markdown(highlight_keywords(job_text), unsafe_allow_html=True)

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("📝 Feedback")

        feedback = st.radio("Was this prediction correct?", ["Yes", "No"], horizontal=True)
        comment = st.text_area("Additional comments (optional)", height=90)

        if st.button("💾 Submit Feedback"):
            save_feedback(job_text,
                          st.session_state.prediction_label,
                          feedback,
                         comment)
            st.success("✅ Thank you for giving feedback!")
# ======================================================
# ADMIN DASHBOARD
# ======================================================
elif st.session_state.page == "admin":

    if not st.session_state.admin_logged:
        st.subheader("🔐 Admin Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
                st.session_state.admin_logged = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")

    else:
        st.subheader("📊 Admin Dashboard")

        # ----------------------------
        # LOAD DATASET
        # ----------------------------
        df = pd.read_csv("fake_job_postings.csv")

        # ----------------------------
        # METRICS
        # ----------------------------
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Jobs", len(df))
        c2.metric("Fake Jobs", df["fraudulent"].sum())
        c3.metric("Fake %", round(df["fraudulent"].mean() * 100, 2))

        st.markdown("---")

        # =================================================
        # 📂 DATASET VIEWER (NEW FEATURE)
        # =================================================
        st.subheader("📂 Dataset Used in This Project")

        show_data = st.checkbox("👁 Show Full Dataset", key="admin_show_dataset")

        if show_data:

            col1, col2 = st.columns(2)

            with col1:
                job_type = st.selectbox(
                    "Filter by Job Type",
                    ["All", "Real Jobs", "Fake Jobs"],
                    key="admin_job_filter"
                )

            with col2:
                search_text = st.text_input(
                    "🔍 Search in Dataset",
                    key="admin_search_text"
                )

            filtered_df = df.copy()

            if job_type == "Real Jobs":
                filtered_df = filtered_df[filtered_df["fraudulent"] == 0]
            elif job_type == "Fake Jobs":
                filtered_df = filtered_df[filtered_df["fraudulent"] == 1]

            if search_text:
                filtered_df = filtered_df[
                    filtered_df.astype(str)
                    .apply(
                        lambda row: row.str.contains(search_text, case=False).any(),
                        axis=1
                    )
                ]

            st.write(f"Showing **{len(filtered_df)}** records")

            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=420
            )

            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Dataset as CSV",
                csv,
                "fake_job_dataset_filtered.csv",
                "text/csv"
            )

        st.markdown("---")

        # =================================================
        # 📝 USER FEEDBACK SECTION (EXISTING)
        # =================================================
        st.markdown("### 📁 User Feedback")

        if os.path.exists("feedback.csv"):
            fb_df = pd.read_csv("feedback.csv")
            st.dataframe(fb_df, use_container_width=True)

            st.markdown("### ⚠️ Clear Feedback Data")
            confirm = st.checkbox(
                "I understand this will permanently delete all feedback data",
                key="confirm_clear_feedback"
            )

            if st.button("🗑 Clear Feedback Data"):
                if confirm:
                    clear_feedback_data()
                    st.success("✅ Feedback data cleared successfully")
                    st.rerun()
                else:
                    st.warning("Please confirm before clearing the data.")
        else:
            st.info("No feedback data available.")
