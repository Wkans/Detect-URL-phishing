"""
train.py
--------
Script chính: huấn luyện toàn bộ pipeline lai (Hybrid CNN-LSTM + Random Forest).

QUY TRÌNH (đúng với kiến trúc đã thiết kế):
1. Đọc & làm sạch dữ liệu, chia train/val/test
2. Nhánh 1: Tokenize URL theo ký tự -> huấn luyện CNN-LSTM end-to-end
            -> trích "deep feature vector" từ lớp Dense áp chót
3. Nhánh 2: Trích đặc trưng thủ công (lexical/domain-based) bằng feature_engineering.py
            -> chuẩn hóa (StandardScaler)
4. Feature Fusion: nối (concatenate) deep features + hand-crafted features
5. Huấn luyện Random Forest trên vector đặc trưng đã hợp nhất
6. Lưu lại: model CNN-LSTM, tokenizer, scaler, Random Forest

CÁCH CHẠY TRÊN PYCHARM:
    - Mở project trong PyCharm, đánh dấu thư mục `src` là "Sources Root"
      (chuột phải vào src -> Mark Directory as -> Sources Root)
    - Chạy trực tiếp file này (Run 'train')
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

from config import (
    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_STATE,
    CNN_LSTM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_N_JOBS
)
from data_preparation import load_and_clean_data, split_data
from char_tokenizer import build_tokenizer, urls_to_sequences, save_tokenizer
from feature_engineering import build_feature_dataframe
from cnn_lstm_model import build_cnn_lstm_model, build_feature_extractor


def main():
    # ===== BƯỚC 1: Dữ liệu =====
    df = load_and_clean_data()
    train_df, val_df, test_df = split_data(df)

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # ===== BƯỚC 2 (Nhánh 1): Tokenize + huấn luyện CNN-LSTM =====
    print("\n===== Huấn luyện nhánh CNN-LSTM (deep features) =====")
    tokenizer = build_tokenizer(train_df["url"])
    save_tokenizer(tokenizer)

    X_train_seq = urls_to_sequences(train_df["url"], tokenizer)
    X_val_seq = urls_to_sequences(val_df["url"], tokenizer)
    X_test_seq = urls_to_sequences(test_df["url"], tokenizer)

    cnn_lstm_model = build_cnn_lstm_model()
    cnn_lstm_model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(CNN_LSTM_MODEL_PATH, monitor="val_loss", save_best_only=True)
    ]

    cnn_lstm_model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Trích deep feature vector cho cả 3 tập train/val/test
    feature_extractor = build_feature_extractor(cnn_lstm_model)
    deep_feat_train = feature_extractor.predict(X_train_seq, batch_size=BATCH_SIZE)
    deep_feat_val = feature_extractor.predict(X_val_seq, batch_size=BATCH_SIZE)
    deep_feat_test = feature_extractor.predict(X_test_seq, batch_size=BATCH_SIZE)
    print("Kích thước deep feature vector:", deep_feat_train.shape)

    # ===== BƯỚC 3 (Nhánh 2): Đặc trưng thủ công =====
    print("\n===== Trích xuất đặc trưng thủ công (hand-crafted features) =====")
    hc_train = build_feature_dataframe(train_df["url"])
    hc_val = build_feature_dataframe(val_df["url"])
    hc_test = build_feature_dataframe(test_df["url"])

    scaler = StandardScaler()
    hc_train_scaled = scaler.fit_transform(hc_train)
    hc_val_scaled = scaler.transform(hc_val)
    hc_test_scaled = scaler.transform(hc_test)
    joblib.dump(scaler, SCALER_PATH)

    # ===== BƯỚC 4: Feature Fusion (nối 2 nhánh) =====
    print("\n===== Hợp nhất đặc trưng (Feature Fusion) =====")
    X_train_fused = np.concatenate([deep_feat_train, hc_train_scaled], axis=1)
    X_val_fused = np.concatenate([deep_feat_val, hc_val_scaled], axis=1)
    X_test_fused = np.concatenate([deep_feat_test, hc_test_scaled], axis=1)
    print("Kích thước vector đặc trưng sau khi fusion:", X_train_fused.shape)

    # Gộp lại train+val để RF học trên nhiều dữ liệu hơn (không bắt buộc,
    # nếu muốn giữ val riêng để tinh chỉnh hyperparameter thì bỏ dòng dưới)
    X_rf_train = np.concatenate([X_train_fused, X_val_fused], axis=0)
    y_rf_train = np.concatenate([y_train, y_val], axis=0)

    # ===== BƯỚC 5: Huấn luyện Random Forest =====
    print("\n===== Huấn luyện Random Forest trên vector hợp nhất =====")
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
        class_weight="balanced"   # hữu ích nếu dữ liệu mất cân bằng lớp
    )
    rf_model.fit(X_rf_train, y_rf_train)
    joblib.dump(rf_model, RF_MODEL_PATH)

    print("\n✅ Đã huấn luyện và lưu xong toàn bộ pipeline:")
    print(f"   - CNN-LSTM model : {CNN_LSTM_MODEL_PATH}")
    print(f"   - Random Forest  : {RF_MODEL_PATH}")
    print(f"   - Scaler         : {SCALER_PATH}")

    # Lưu tập test đã fusion để evaluate.py dùng lại, không cần train lại
    np.savez(
        "../data/test_fused_features.npz",
        X_test=X_test_fused, y_test=y_test
    )
    print("Đã lưu đặc trưng tập test để phục vụ evaluate.py")


if __name__ == "__main__":
    main()
