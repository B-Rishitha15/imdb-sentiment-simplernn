# IMDB Movie Review Sentiment Analysis (SimpleRNN)

A Streamlit web app that classifies a movie review as **Positive** or **Negative** using a SimpleRNN model trained on the IMDB dataset.

## Demo

Live app:https://imdb-sentiment-simplernn-7pabjbrfsmt9cfrsh4smzy.streamlit.app

## Model

| Item | Value |
|---|---|
| Dataset | IMDB (25,000 train / 25,000 test reviews) |
| Vocabulary size | 10,000 most frequent words |
| Sequence length | 500 (pre-padded) |
| Architecture | Embedding(128) -> SimpleRNN(128, tanh, dropout 0.2) -> Dense(1, sigmoid) |
| Optimizer | Adam (clipnorm=1.0) |
| Loss | Binary cross-entropy |
| Early stopping | Yes (patience 3, best weights restored) |
| Test accuracy | **78.31%** |

## Project structure

```
.
├── main.py                # Streamlit app
├── simplernn.ipynb        # Training notebook
├── prediction.ipynb       # Test the saved model
├── simple_rnn_imdb.h5     # Trained model
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run main.py
```

Python 3.11 with TensorFlow 2.15 is recommended.

## How it works

1. The review is lowercased and split into words (punctuation removed).
2. Each word is mapped to its IMDB word index (+3 offset, as in Keras). Unknown words and words outside the top 10,000 map to the OOV token.
3. A start token is added and the sequence is padded to 500 tokens.
4. The model outputs a score between 0 and 1. A score above 0.5 is Positive, otherwise Negative.

## Known limitations

- **Accuracy is limited.** A SimpleRNN reaches roughly 78-83% on IMDB (results vary between training runs). LSTM/GRU models typically reach 85-88%.
- **Negation is a weak spot.** Reviews like "the movie was scary, i did not like it" can be classified wrongly (the score stays close to 0.5), likely because SimpleRNN struggles to carry information across a long sequence.
- **Short inputs are less reliable.** The model was trained on full-length reviews (about 230 words on average). Very short reviews give less confident scores.
- Scores close to 0.5 mean the model is uncertain.

## Possible improvements

- Replace SimpleRNN with LSTM or GRU.
- Use Bidirectional layers.
- Use pretrained embeddings (GloVe) or a transformer model.

## Tech stack
