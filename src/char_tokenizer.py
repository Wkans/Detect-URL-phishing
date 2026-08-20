"""
char_tokenizer.py
------------------
Chuyển URL (chuỗi thô) thành chuỗi số nguyên ở cấp độ ký tự (character-level),
phục vụ cho embedding layer đầu vào của mô hình CNN-LSTM.
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

from config import MAX_URL_LENGTH, VOCAB_SIZE, TOKENIZER_PATH


def build_tokenizer(urls):
    """Xây dựng bộ char-tokenizer dựa trên tập URL huấn luyện."""
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        char_level=True,      # QUAN TRỌNG: tokenize theo từng ký tự, không theo từ
        oov_token="<OOV>",
        lower=True,
    )
    tokenizer.fit_on_texts(urls)
    return tokenizer


def urls_to_sequences(urls, tokenizer):
    """Chuyển danh sách URL thành ma trận số nguyên đã pad về cùng độ dài."""
    sequences = tokenizer.texts_to_sequences(urls)
    padded = pad_sequences(
        sequences, maxlen=MAX_URL_LENGTH, padding="post", truncating="post"
    )
    return padded


def save_tokenizer(tokenizer, path=TOKENIZER_PATH):
    joblib.dump(tokenizer, path)
    print(f"Đã lưu tokenizer tại: {path}")


def load_tokenizer(path=TOKENIZER_PATH):
    return joblib.load(path)


if __name__ == "__main__":
    test_urls = ["http://google.com", "http://phishing-site.tk/login"]
    tok = build_tokenizer(test_urls)
    seq = urls_to_sequences(test_urls, tok)
    print("Vocab size thực tế:", len(tok.word_index))
    print("Shape sau khi pad:", seq.shape)
    print(seq)
