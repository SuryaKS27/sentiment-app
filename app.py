import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🧠 Sentiment Analysis with BERT")
st.write("Enter some text and see what the model thinks about the sentiment.")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

user_input = st.text_area("Enter your text here", "I love this product!")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result = classifier(user_input)[0]
        st.write(f"**Label:** {result['label']}")
        st.write(f"**Confidence:** {round(result['score']*100, 2)}%")
