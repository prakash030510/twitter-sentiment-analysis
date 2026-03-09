import streamlit as st
import joblib
import re
import matplotlib.pyplot as plt
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------- LOAD MODEL ----------------
model = joblib.load(r"C:\Twitter Analysis project\model\sentiment_model.pkl")
vectorizer = joblib.load(r"C:\Twitter Analysis project\model\vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'[^a-zA-Z]',' ',text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- UI ----------------
st.title("🐦 Twitter Sentiment Analysis & Brand Monitoring System")
st.write("AI system that predicts tweet sentiment, stores history and shows analytics dashboard")

# ---------------- FILE PATH ----------------
history_path = r"C:\Twitter Analysis project\app\history.csv"

# create history file if not exists
if not os.path.exists(history_path):
    pd.DataFrame({"Tweet": [], "Sentiment": []}).to_csv(history_path, index=False)

# ---------------- TWEET PREDICTION ----------------
st.header("✏️ Tweet Sentiment Prediction")
tweet = st.text_area("Enter your tweet here:")

if "pos" not in st.session_state:
    st.session_state.pos = 0
if "neg" not in st.session_state:
    st.session_state.neg = 0

if st.button("Predict Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet")
    else:
        cleaned = clean_text(tweet)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            result = "Positive"
            st.success("Positive 🙂")
            st.session_state.pos += 1
        else:
            result = "Negative"
            st.error("Negative 😡")
            st.session_state.neg += 1

        # save history
        pd.DataFrame([[tweet, result]], columns=["Tweet", "Sentiment"])\
            .to_csv(history_path, mode='a', header=False, index=False)

# ---------------- OVERALL DASHBOARD ----------------
if st.session_state.pos + st.session_state.neg > 0:
    st.subheader("📊 Overall Sentiment Dashboard")

    labels = ['Positive', 'Negative']
    sizes = [st.session_state.pos, st.session_state.neg]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# ---------------- BRAND ANALYSIS ----------------
st.header("🏢 Brand Reputation Checker")

brand = st.text_input("Enter Brand Name (Example: Amazon, Swiggy, iPhone)")

if st.button("Analyze Brand"):
    if brand.strip() == "":
        st.warning("Please enter a brand name")
    else:
        sample_tweets = [
            f"I love {brand} service",
            f"{brand} delivery is very fast",
            f"{brand} customer support is bad",
            f"I hate {brand} packaging",
            f"{brand} product quality is amazing",
            f"{brand} is very disappointing",
            f"{brand} support team helped me a lot",
            f"{brand} refund process is slow",
            f"{brand} app interface is very smooth",
            f"{brand} experience was terrible"
        ]

        pos = 0
        neg = 0

        for t in sample_tweets:
            cleaned = clean_text(t)
            vector = vectorizer.transform([cleaned])
            pred = model.predict(vector)[0]

            if pred == 1:
                pos += 1
            else:
                neg += 1

        st.write(f"Positive Mentions: {pos}")
        st.write(f"Negative Mentions: {neg}")

        if pos > neg:
            st.success(f"Overall Public Opinion on {brand} is POSITIVE 👍")
        else:
            st.error(f"Overall Public Opinion on {brand} is NEGATIVE 👎")

        # Brand Pie Chart
        st.subheader("📊 Brand Sentiment Distribution")

        fig2, ax2 = plt.subplots()
        ax2.pie([pos, neg], labels=['Positive','Negative'], autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)

# ---------------- HISTORY TABLE ----------------
st.subheader("📜 Prediction History")

history_df = pd.read_csv(history_path)
st.dataframe(history_df.tail(10))
