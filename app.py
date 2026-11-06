"""Web UI for the trained TN4 phishing-URL model (Streamlit).
Run with: .\\.venv\\Scripts\\python.exe -m streamlit run app.py
Never visits the URLs it is asked about; only analyzes the string itself.
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from common import encode_urls
from features import feature_matrix, parse_url, normalize_url, LEXICAL_NAMES
from model import get_extractor
from typosquat import apply_rule, check_url

st.set_page_config(page_title='Phát hiện URL giả mạo', page_icon='🔎', layout='centered')


def list_model_dirs():
    root = Path('runs')
    if not root.exists():
        return []
    natural = lambda n: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', n)]
    return sorted((p.name for p in root.iterdir() if (p / 'config.json').exists()), key=natural, reverse=True)


@st.cache_resource(show_spinner='Đang nạp mô hình...')
def load_bundle(model_dir: str):
    root = Path('runs') / model_dir
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    cfg, vocab = read('config.json'), read('vocab.json')
    schema = read('feature_schema.json')['raw']
    if schema != LEXICAL_NAMES:
        raise ValueError(
            'Mô hình này dùng bộ đặc trưng khác với features.py hiện tại. '
            'Hãy train lại bằng train.py hiện tại trong một thư mục --out mới.')
    network = keras.models.load_model(root / 'cnn_lstm.keras', compile=False)
    extractor = get_extractor(network)
    imputer = joblib.load(root / 'hand_imputer.joblib')
    rf = joblib.load(root / 'TN4_Hybrid_RF.joblib')
    threshold = read('thresholds.json')['TN4_Hybrid_RF']
    return dict(cfg=cfg, vocab=vocab, extractor=extractor, imputer=imputer, rf=rf, threshold=threshold)


def predict_urls(urls: list[str], bundle: dict) -> pd.DataFrame:
    cfg, vocab = bundle['cfg'], bundle['vocab']
    urls = [str(u).strip() for u in urls]
    model_urls = [normalize_url(u) for u in urls] if cfg.get('url_normalization') else urls
    bad = []
    for i, u in enumerate(model_urls):
        try:
            parse_url(u)
        except ValueError:
            bad.append(urls[i])
    if bad:
        raise ValueError('Không phân tích được cú pháp URL: ' + '; '.join(bad[:5])
                          + (f' (và {len(bad) - 5} URL khác)' if len(bad) > 5 else ''))
    results = []
    for start in range(0, len(urls), 256):
        batch = urls[start:start + 256]
        mbatch = model_urls[start:start + 256]
        x = encode_urls(mbatch, vocab, cfg['max_len'])
        h, _ = feature_matrix(mbatch)
        d = bundle['extractor'].predict(x, verbose=0)
        p = bundle['rf'].predict_proba(np.concatenate([d, bundle['imputer'].transform(h)], axis=1))[:, 1]
        for u, mu, score, row in zip(batch, mbatch, p, h):
            hit = check_url(u)
            final = apply_rule(float(score), hit)
            results.append({
                'url': u,
                'model_score': float(score),
                'phishing_score': final,
                'threshold': bundle['threshold'],
                'label': int(final >= bundle['threshold']),
                'class_name': 'phishing' if final >= bundle['threshold'] else 'legitimate',
                'truncated': len(mu) > cfg['max_len'],
                'missing_hand_features': int(np.isnan(row).sum()),
                'rule': hit['rule'] if hit else '',
                'rule_reason': hit['reason'] if hit else '',
            })
    return pd.DataFrame(results)


# ---- Sidebar: model selection ----
st.sidebar.header('Mô hình')
dirs = list_model_dirs()
if not dirs:
    st.sidebar.error('Không tìm thấy thư mục nào trong runs/. Hãy train trước bằng train.py.')
    st.stop()
model_dir = st.sidebar.selectbox('Thư mục kết quả (runs/...)', dirs, index=0)
try:
    bundle = load_bundle(model_dir)
except Exception as e:
    st.sidebar.error(f'Lỗi khi nạp mô hình: {e}')
    st.stop()
st.sidebar.caption(f"Ngưỡng quyết định: {bundle['threshold']:.3f}")
st.sidebar.caption(f"max_len: {bundle['cfg']['max_len']}")

st.title('🔎 Phát hiện URL giả mạo')
st.caption('Mô hình CNN-LSTM kết hợp Random Forest. Chỉ phân tích chuỗi URL, không truy cập địa chỉ được nhập.')

tab_single, tab_batch = st.tabs(['Kiểm tra 1 URL', 'Kiểm tra nhiều URL (CSV)'])

with tab_single:
    url = st.text_input('Nhập URL cần kiểm tra', placeholder='https://example.com/login')
    if st.button('Kiểm tra', type='primary', disabled=not url.strip()):
        try:
            df = predict_urls([url], bundle)
        except ValueError as e:
            st.error(str(e))
        else:
            row = df.iloc[0]
            score = row.phishing_score
            if row.label == 1:
                st.error(f'⚠️ GIẢ MẠO (phishing) — điểm {score:.4f} (ngưỡng {row.threshold:.3f})')
            else:
                st.success(f'✅ HỢP LỆ (legitimate) — điểm {score:.4f} (ngưỡng {row.threshold:.3f})')
            st.progress(min(max(score, 0.0), 1.0))
            if row.rule:
                st.warning(f'Luật {row.rule} kích hoạt: {row.rule_reason}. '
                           f'Điểm gốc của mô hình: {row.model_score:.4f}.')
            c1, c2 = st.columns(2)
            c1.metric('Bị cắt bớt (>max_len)', 'Có' if row.truncated else 'Không')
            c2.metric('Đặc trưng thủ công thiếu', int(row.missing_hand_features))
            

with tab_batch:
    st.write('Tải lên CSV có cột **url** (có thể có cột khác, sẽ bị bỏ qua).')
    up = st.file_uploader('Chọn file CSV', type=['csv'])
    if up is not None:
        try:
            src = pd.read_csv(up, dtype={'url': 'string'})
            urls = src.url.fillna('').tolist()
        except Exception as e:
            st.error(f'Không đọc được CSV: {e}')
        else:
            if st.button('Chạy dự đoán', type='primary'):
                with st.spinner(f'Đang dự đoán {len(urls)} URL...'):
                    try:
                        df = predict_urls(urls, bundle)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.success(f'Xong: {len(df)} URL, {int(df.label.sum())} bị gắn nhãn phishing.')
                        st.dataframe(df, use_container_width=True)
                        st.download_button('Tải kết quả CSV', df.to_csv(index=False).encode('utf-8'),
                                            file_name='predictions.csv', mime='text/csv')

st.divider()
