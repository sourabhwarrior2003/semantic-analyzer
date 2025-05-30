import sys
import os
import streamlit as st                             

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_text_analyzer.utils.preprocess import clean_text
from wordcloud import WordCloud                    

import matplotlib.pyplot as plt                    
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Streamlit page setup
st.set_page_config(page_title="Semantic Text Analyzer")

st.title("🧠 Semantic Text Analyzer")

# Text input
text_input = st.text_area("Paste your paragraph here:", height=200)

# Option to toggle stemming
do_stemming = st.checkbox("Enable Stemming", value=False)

if st.button("Analyze Text"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input, do_stemming=do_stemming)

        st.subheader("🧼 Cleaned Text")
        st.write(cleaned)

        if cleaned.strip() == "":
            st.warning("The cleaned text is empty after preprocessing. Please enter different text.")
        else:
            # TF-IDF Analysis
            tfidf = TfidfVectorizer()
            X = tfidf.fit_transform([cleaned])
            df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())

            st.subheader("📊 Top TF-IDF Scores")
            top_words = df.iloc[0].sort_values(ascending=False).head(10)
            st.bar_chart(top_words)

            st.subheader("☁️ Word Cloud")
            wc = WordCloud(width=800, height=400, background_color='white').generate(cleaned)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
