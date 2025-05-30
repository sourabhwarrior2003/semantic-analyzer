# 🧠 Semantic Text Analyzer

This is a Streamlit web app that performs semantic analysis of text using spaCy, NLTK, TF-IDF, and WordCloud.

## ✨ Features
- Cleans and lemmatizes text
- Extracts top TF-IDF features
- Generates word clouds

## 🚀 How to Run

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
streamlit run app.py
```

## 📂 Structure

- `app.py`: Main Streamlit interface
- `utils/preprocess.py`: NLP preprocessing functions
- `requirements.txt`: Python dependencies
- `README.md`: Project info
