"""
config.py
---------
File cấu hình tập trung cho toàn bộ đồ án.
Chỉnh sửa các tham số ở đây thay vì rải rác trong nhiều file.
"""

import os

# ===== ĐƯỜNG DẪN =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "urls_raw.csv")          # dataset gốc: url,label
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "urls_processed.csv")  # sau khi trích đặc trưng thủ công

CNN_LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_model.keras")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "char_tokenizer.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")

# ===== THAM SỐ XỬ LÝ URL (NHÁNH CHUỖI KÝ TỰ) =====
MAX_URL_LENGTH =150        # cắt/pad URL về độ dài cố định 200 ký tự
CHAR_EMBEDDING_DIM = 64     # số chiều embedding cho mỗi ký tự
VOCAB_SIZE = 128            # số ký tự khác nhau tối đa (ASCII printable ~ đủ dùng)

# ===== THAM SỐ MÔ HÌNH CNN-LSTM =====
CNN_FILTERS = 128
CNN_KERNEL_SIZE = 5
LSTM_UNITS = 64
DROPOUT_RATE = 0.5
DENSE_UNITS = 64            # lớp dense cuối cùng trước khi lấy làm "deep feature vector"

# ===== THAM SỐ HUẤN LUYỆN =====
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
RANDOM_STATE = 42

# ===== THAM SỐ RANDOM FOREST =====
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_N_JOBS = -1

os.makedirs(MODEL_DIR, exist_ok=True)
