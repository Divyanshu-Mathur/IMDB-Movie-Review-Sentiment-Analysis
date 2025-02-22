import  numpy  as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing  import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense,Embedding
from tensorflow.keras.models import load_model
import streamlit as slt

word_index = imdb.get_word_index()
rev_word_index = {v:k for k,v in word_index.items()}

model=load_model('model.h5')
def decode(review):
    review = review.flatten().tolist()
    return " ".join([rev_word_index.get(i-3,"?") for i in review])

def preprocess_text(text):
    words= text.lower().split()
    review  = [word_index.get(word,1)+3 for word in words]
    padding = sequence.pad_sequences([review],maxlen=500)
    return padding


def predict_sentiment(review):
    preprocess = preprocess_text(review)
    print(decode(preprocess))
    pred = model.predict(preprocess)
    sentiment = 'Positive' if pred[0][0]> 0.5 else 'Negative'
    return sentiment,pred[0][0]



slt.title('IMDB Movie Review Sentiment Analysis')
slt.write('Enter a movie review to classify it as positive or negative')

user_input = slt.text_area('Movie Review')

if slt.button('Classify'):
    sentiment,prediction = predict_sentiment(user_input)
    slt.write(f"Sentiment : {sentiment}")
    slt.write(f"Score : {prediction}")
else:
    slt.write("Please enter a movie review")
    