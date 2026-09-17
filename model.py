"""Character Embedding -> Conv1D -> masked LSTM -> deep vector -> sigmoid."""
import tensorflow as tf
from tensorflow import keras


def build_model(vocab_size, max_len=256, embedding_dim=64, filters=64,
                kernel_size=5, lstm_units=64, deep_dim=64):
    ids = keras.Input(shape=(max_len,), dtype='int32', name='url_tokens')
    valid = keras.ops.not_equal(ids, 0)
    x = keras.layers.Embedding(vocab_size, embedding_dim, name='character_embedding')(ids)
    # Zero padded embeddings before convolution; pass the true token mask to LSTM.
    mask = keras.ops.expand_dims(keras.ops.cast(valid, 'float32'), -1)
    x = keras.layers.Multiply()([x, mask])
    x = keras.layers.Conv1D(filters, kernel_size, padding='same', activation='relu', name='local_patterns')(x)
    x = keras.layers.LSTM(lstm_units, name='sequence_context')(x, mask=valid)
    deep = keras.layers.Dense(deep_dim, activation='relu', name='deep_features')(x)
    x = keras.layers.Dropout(.3)(deep)
    out = keras.layers.Dense(1, activation='sigmoid', name='phishing_probability')(x)
    model = keras.Model(ids, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='roc_auc')])
    return model


def get_extractor(model):
    return keras.Model(model.input, model.get_layer('deep_features').output)
