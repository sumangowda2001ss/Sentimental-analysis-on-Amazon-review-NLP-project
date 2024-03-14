# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:17:16 2024

@author: suman
"""

import streamlit as st
import re
import pandas as pd
from io import BytesIO
import base64
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

STOPWORDS = set(stopwords.words("english"))

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph

def main():
    st.title("Sentiment Analysis Web Application")
    st.sidebar.title("Options")

    selected_option = st.sidebar.radio("Choose an option", ("Single Text Prediction", "Bulk Prediction"))

    if selected_option == "Single Text Prediction":
        st.subheader("Single Text Prediction")
        text_input = st.text_input("Enter your text here:")
        if st.button("Predict"):
            predictor = pickle.load(open("model_xgb.pkl", "rb"))
            scaler = pickle.load(open("scaler.pkl", "rb"))
            cv = pickle.load(open("countVectorizer (1).pkl", "rb"))
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            st.write(f"Predicted sentiment: {predicted_sentiment}")

    elif selected_option == "Bulk Prediction":
        st.subheader("Bulk Prediction")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            predictor = pickle.load(open("model_xgb.pkl", "rb"))
            scaler = pickle.load(open("scaler.pkl", "rb"))
            cv = pickle.load(open("countVectorizer (1).pkl", "rb"))
            corpus = []
            stemmer = PorterStemmer()
            for i in range(0, data.shape[0]):
                review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
                review = review.lower().split()
                review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
                review = " ".join(review)
                corpus.append(review)

            X_prediction = cv.transform(corpus).toarray()
            X_prediction_scl = scaler.transform(X_prediction)
            y_predictions = predictor.predict_proba(X_prediction_scl)
            y_predictions = y_predictions.argmax(axis=1)
            y_predictions = list(map(sentiment_mapping, y_predictions))

            data["Predicted sentiment"] = y_predictions
            st.write(data)

            graph = get_distribution_graph(data)
            st.image(graph, caption='Sentiment Distribution', use_column_width=True)

def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"

if __name__ == "__main__":
    main()
    

