"""REST API for the trained TN4 phishing-URL model (FastAPI).

Run with:  .\\.venv\\Scripts\\python.exe -m uvicorn api:app --reload
Interactive docs (a built-in web UI to try it from the browser): http://127.0.0.1:8000/docs

Never visits the URLs it is asked about; only analyzes the string itself.
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow import keras

from common import encode_urls
from features import feature_matrix, parse_url, normalize_url, LEXICAL_NAMES
from model import get_extractor
from typosquat import apply_rule, check_url

RUNS_DIR = Path('runs')
# Default model: the --model-dir you want the API to use when the caller does not pick one.
DEFAULT_MODEL_DIR = os.environ.get('APP_MODEL_DIR', 'url_only_v10')

app = FastAPI(
    title='Phát hiện URL giả mạo API',
    description='CNN-LSTM kết hợp Random Forest. Chỉ phân tích chuỗi URL, không truy cập địa chỉ được gửi lên.',
    version='1.0.0',
)

_bundles: dict[str, dict] = {}  # model_dir -> loaded components, cached after first use


def available_models() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    natural = lambda n: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', n)]
    return sorted((p.name for p in RUNS_DIR.iterdir() if (p / 'config.json').exists()), key=natural, reverse=True)


def load_bundle(model_dir: str) -> dict:
    if model_dir in _bundles:
        return _bundles[model_dir]
    root = RUNS_DIR / model_dir
    if not root.exists():
        raise HTTPException(404, f'Không tìm thấy mô hình: {model_dir}. Xem GET /models.')
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    try:
        cfg, vocab = read('config.json'), read('vocab.json')
        schema = read('feature_schema.json')['raw']
    except FileNotFoundError as e:
        raise HTTPException(500, f'Thư mục mô hình thiếu file: {e}')
    if schema != LEXICAL_NAMES:
        raise HTTPException(500, 'Mô hình dùng bộ đặc trưng khác với features.py hiện tại. Cần train lại.')
    network = keras.models.load_model(root / 'cnn_lstm.keras', compile=False)
    bundle = dict(
        cfg=cfg, vocab=vocab,
        extractor=get_extractor(network),
        imputer=joblib.load(root / 'hand_imputer.joblib'),
        rf=joblib.load(root / 'TN4_Hybrid_RF.joblib'),
        threshold=read('thresholds.json')['TN4_Hybrid_RF'],
    )
    _bundles[model_dir] = bundle
    return bundle


def run_predictions(urls: list[str], bundle: dict) -> list[dict]:
    cfg, vocab = bundle['cfg'], bundle['vocab']
    urls = [str(u).strip() for u in urls]
    model_urls = [normalize_url(u) for u in urls] if cfg.get('url_normalization') else urls
    for i, u in enumerate(model_urls):
        try:
            parse_url(u)
        except ValueError:
            raise HTTPException(422, f'Không phân tích được cú pháp URL: {urls[i]!r}')
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
    return results


@app.on_event('startup')
def warm_default_model():
    if DEFAULT_MODEL_DIR in available_models():
        load_bundle(DEFAULT_MODEL_DIR)


class PredictRequest(BaseModel):
    url: str = Field(..., examples=['https://example.com/login'])
    model_dir: Optional[str] = Field(None, description='Tên thư mục trong runs/. Mặc định dùng model đã nạp sẵn.')


class BatchPredictRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=5000)
    model_dir: Optional[str] = None


class Prediction(BaseModel):
    url: str
    model_score: float = Field(..., description='Điểm gốc của mô hình, chưa qua luật typosquatting.')
    phishing_score: float = Field(..., description='Điểm cuối: bằng model_score, hoặc được nâng lên nếu luật kích hoạt.')
    rule: str = Field('', description="Luật kích hoạt: 'homoglyph', 'edit1', 'subdomain', 'brand_token' hoặc rỗng.")
    rule_reason: str = ''
    threshold: float
    label: int
    class_name: str
    truncated: bool
    missing_hand_features: int


@app.get('/health')
def health():
    return {'status': 'ok', 'default_model': DEFAULT_MODEL_DIR, 'loaded_models': list(_bundles)}


@app.get('/models')
def models():
    return {'available': available_models(), 'default': DEFAULT_MODEL_DIR}


@app.post('/predict', response_model=Prediction)
def predict(req: PredictRequest):
    bundle = load_bundle(req.model_dir or DEFAULT_MODEL_DIR)
    return run_predictions([req.url], bundle)[0]


@app.post('/predict/batch', response_model=list[Prediction])
def predict_batch(req: BatchPredictRequest):
    bundle = load_bundle(req.model_dir or DEFAULT_MODEL_DIR)
    return run_predictions(req.urls, bundle)
