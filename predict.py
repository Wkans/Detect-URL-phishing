"""Predict with saved TN4 using URL strings only; never visits submitted URLs."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from common import encode_urls
from features import feature_matrix, parse_url, LEXICAL_NAMES
from model import get_extractor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--url')
    source.add_argument('--csv', help='CSV with url column')
    ap.add_argument('--out', default='predictions.csv')
    args = ap.parse_args()
    root = Path(args.model_dir)
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    cfg, vocab = read('config.json'), read('vocab.json')
    if read('feature_schema.json')['raw'] != LEXICAL_NAMES:
        ap.error('This model uses a different feature set. Retrain with the current train.py in a new output directory.')
    urls = [args.url] if args.url else pd.read_csv(args.csv, dtype={'url': 'string'}).url.fillna('').tolist()
    if not urls:
        raise ValueError('No URLs supplied')
    urls = [str(u).strip() for u in urls]
    for u in urls:
        parse_url(u)
    network = keras.models.load_model(root / 'cnn_lstm.keras', compile=False)
    extractor = get_extractor(network)
    imputer = joblib.load(root / 'hand_imputer.joblib')
    rf = joblib.load(root / 'TN4_Hybrid_RF.joblib')
    threshold = read('thresholds.json')['TN4_Hybrid_RF']
    results = []
    for start in range(0, len(urls), 256):
        batch = urls[start:start + 256]
        x = encode_urls(batch, vocab, cfg['max_len'])
        h, names = feature_matrix(batch)
        if names != read('feature_schema.json')['raw']:
            raise ValueError('Feature schema does not match training')
        d = extractor.predict(x, verbose=0)
        p = rf.predict_proba(np.concatenate([d, imputer.transform(h)], axis=1))[:, 1]
        for u, score, row in zip(batch, p, h):
            results.append({'url': u, 'phishing_score': float(score), 'threshold': threshold,
                            'label': int(score >= threshold),
                            'class_name': 'phishing' if score >= threshold else 'legitimate',
                            'truncated': len(u) > cfg['max_len'],
                            'missing_hand_features': int(np.isnan(row).sum())})
    result = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.head(20).to_string(index=False))
    print('Scores are model outputs, not calibrated real-world probabilities.')


if __name__ == '__main__':
    main()
