import re
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

MAX_FEATURES = 10000
MAX_LEN = 500


@st.cache_resource
def load_assets():
    word_index = imdb.get_word_index()
    model = load_model('simple_rnn_imdb.h5')
    return word_index, model


word_index, model = load_assets()


def preprocess_text(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    encoded = [1]  # start token, same as training data
    for w in words:
        idx = word_index.get(w)
        if idx is None or idx + 3 >= MAX_FEATURES:
            encoded.append(2)  # out-of-vocabulary
        else:
            encoded.append(idx + 3)
    return sequence.pad_sequences([encoded], maxlen=MAX_LEN)


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review', key='review_input')

if st.button('Classify', key='classify_btn'):
    if not user_input.strip():
        st.warning('Please enter a movie review.')
    else:
        score = float(model.predict(preprocess_text(user_input))[0][0])
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score:.4f}')