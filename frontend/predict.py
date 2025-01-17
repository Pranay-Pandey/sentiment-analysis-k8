import streamlit as st
from transformers import pipeline

sentimentclassifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

def sentiment_analysis(text):
    return sentimentclassifier(text)

st.title('Sentiment Analysis')

# Create a text area for user input
user_input = st.text_area('Enter some text')

# Create a button to run the model
if st.button('Run Model'):
    # Use the predict function to run the model
    prediction = sentiment_analysis(user_input)
    score = prediction[0]['score']
    label = prediction[0]['label']
    st.write(f'Text: {user_input}')
    st.write(f'Sentiment: {label}')
    st.write(f'Score: {score}')