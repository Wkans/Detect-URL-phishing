"""
predict.py
----------
Dùng để demo: nhập vào MỘT URL bất kỳ, pipeline sẽ tự động:
1. Tokenize + đưa qua CNN-LSTM đã train -> lấy deep feature vector
2. Trích đặc trưng thủ công + chuẩn hóa bằng scaler đã lưu
3. Ghép 2 vector -> đưa vào Random Forest -> in kết quả dự đoán

CHẠY: python predict.py "http://example-suspicious-site.tk/login.php"
"""

import sys
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from config import (
    CNN_LSTM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH, TOKENIZER_PATH
)
from char_tokenizer import urls_to_sequences, load_tokenizer
from feature_engineering import build_feature_dataframe
from cnn_lstm_model import build_feature_extractor
import sys

def predict_url(url: str):
    # Load các thành phần đã lưu
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    cnn_lstm_model = load_model(CNN_LSTM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    feature_extractor = build_feature_extractor(cnn_lstm_model)

    # Nhánh 1: deep features
    seq = urls_to_sequences([url], tokenizer)
    deep_feat = feature_extractor.predict(seq)

    # Nhánh 2: hand-crafted features
    hc_feat = build_feature_dataframe([url])
    hc_feat_scaled = scaler.transform(hc_feat)

    # Fusion
    fused = np.concatenate([deep_feat, hc_feat_scaled], axis=1)

    # Dự đoán
    pred_label = rf_model.predict(fused)[0]
    pred_proba = rf_model.predict_proba(fused)[0][1]

    label_text = "🚨 PHISHING" if pred_label == 1 else "✅ AN TOÀN (Benign)"
    print(f"\nURL: {url}")
    print(f"Kết quả: {label_text}")
    print(f"Xác suất phishing: {pred_proba:.4f}")
    return pred_label, pred_proba



def main():
    if len(sys.argv) > 1:
        # Nếu có truyền tham số từ dòng lệnh, xử lý URL đó trước
        input_url = sys.argv[1]
        predict_url(input_url)

    # Sau đó liên tục yêu cầu nhập URL mới
    while True:
        input_url = input("Nhập URL cần kiểm tra (hoặc gõ 'exit' để thoát): ")
        if input_url.lower() in ["exit", "quit"]:
            break
        if input_url.strip():  # Kiểm tra URL không bị rỗng
            predict_url(input_url)

if __name__ == "__main__":
    main()