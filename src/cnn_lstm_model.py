"""
cnn_lstm_model.py
------------------
Định nghĩa kiến trúc CNN-LSTM cho nhánh học đặc trưng tự động (deep features)
từ chuỗi ký tự URL thô.

Kiến trúc:
Input (chuỗi số nguyên) -> Embedding -> Conv1D -> MaxPooling1D
    -> LSTM -> Dense (deep feature vector) -> Dense output (sigmoid, để huấn luyện end-to-end)

Sau khi huấn luyện xong mô hình theo kiểu end-to-end (để CNN-LSTM học được
biểu diễn tốt), ta sẽ "cắt" model tại lớp Dense feature để lấy ra vector
đặc trưng sâu, dùng làm input cho Random Forest ở bước feature fusion.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, LSTM,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

from config import (
    MAX_URL_LENGTH, VOCAB_SIZE, CHAR_EMBEDDING_DIM,
    CNN_FILTERS, CNN_KERNEL_SIZE, LSTM_UNITS,
    DROPOUT_RATE, DENSE_UNITS, LEARNING_RATE
)


def build_cnn_lstm_model():
    """Xây dựng model CNN-LSTM end-to-end (dùng để huấn luyện ban đầu)."""
    inputs = Input(shape=(MAX_URL_LENGTH,), name="url_char_input")

    x = Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=CHAR_EMBEDDING_DIM,
        input_length=MAX_URL_LENGTH,
        name="char_embedding"
    )(inputs)

    # Nhánh CNN: bắt các mẫu n-gram ký tự cục bộ (ví dụ "php?", ".tk/", "login")
    x = Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE,
               activation="relu", padding="same", name="conv1d")(x)
    x = MaxPooling1D(pool_size=2, name="maxpool")(x)
    x = BatchNormalization()(x)

    # Nhánh LSTM: bắt phụ thuộc tuần tự dài hạn trong chuỗi ký tự
    x = LSTM(LSTM_UNITS, return_sequences=False, name="lstm")(x)
    x = Dropout(DROPOUT_RATE)(x)

    # Lớp Dense này chính là "deep feature vector" sẽ được trích ra sau này
    deep_features = Dense(DENSE_UNITS, activation="relu", name="deep_feature_layer")(x)
    x = Dropout(DROPOUT_RATE / 2)(deep_features)

    # Output tạm để huấn luyện end-to-end (bài toán phân loại nhị phân)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CNN_LSTM_PhishingDetector")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )
    return model


def build_feature_extractor(trained_model):
    """
    Từ model CNN-LSTM đã huấn luyện xong, tạo ra một model phụ
    dùng để trích xuất vector đặc trưng tại lớp 'deep_feature_layer'.
    Đây chính là cách lấy 'deep features' để đưa vào Random Forest.
    """
    feature_extractor = Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer("deep_feature_layer").output,
        name="DeepFeatureExtractor"
    )
    return feature_extractor


if __name__ == "__main__":
    model = build_cnn_lstm_model()
    model.summary()
