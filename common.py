"""Data validation, grouped splits, character vocabulary and evaluation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from features import canonical_key, domain_of


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_data(path):
    df = pd.read_csv(path, dtype={'url': 'string'}, encoding='utf-8-sig')
    if not {'url', 'label'} <= set(df.columns):
        raise ValueError('CSV must have columns url,label (0=legitimate, 1=phishing)')
    df = df[['url', 'label']].copy()
    df['source_row'] = np.arange(len(df)) + 2
    report = {'input_rows': len(df)}
    labels = pd.to_numeric(df.label, errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError('Every label must be 0 or 1; fix missing/invalid labels first')
    df['label'] = labels.astype(int)
    df['url'] = df.url.fillna('').str.strip()
    keys, domains = [], []
    for u in df.url:
        try:
            keys.append(repr(canonical_key(u)))
            domains.append(domain_of(u))
        except ValueError:
            keys.append(None)
            domains.append(None)
    df['key'], df['domain'] = keys, domains
    report['invalid_url_rows_removed'] = int(df.key.isna().sum())
    df = df.dropna(subset=['key']).copy()
    conflicts = df.groupby('key').label.nunique()
    bad = set(conflicts[conflicts > 1].index)
    report['conflicting_rows_removed'] = int(df.key.isin(bad).sum())
    df = df[~df.key.isin(bad)].copy()
    before = len(df)
    df = df.drop_duplicates('key').reset_index(drop=True)
    report['duplicate_rows_removed'] = before - len(df)
    report['clean_rows'] = len(df)
    report['label_counts'] = {str(k): int(v) for k, v in df.label.value_counts().items()}
    if df.label.nunique() != 2:
        raise ValueError('Both classes are required after cleaning')
    return df, report


def split_data(df, mode, seed):
    idx = np.arange(len(df))
    if mode == 'random':
        trv, te = train_test_split(idx, test_size=.2, stratify=df.label, random_state=seed)
        tr, va = train_test_split(trv, test_size=.25, stratify=df.iloc[trv].label, random_state=seed)
    else:
        # Approximate 60/20/20 by grouped folds. Exact proportions are not guaranteed.
        if df.domain.nunique() < 10:
            raise ValueError('Domain split needs >=10 distinct domains; add data or use --split random for a smoke test')
        trv, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, df.label, df.domain))
        sub = df.iloc[trv]
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(trv, sub.label, sub.domain))
        tr, va = trv[a], trv[b]
    splits = {'train': tr, 'val': va, 'test': te}
    for name, ix in splits.items():
        if df.iloc[ix].label.nunique() != 2:
            raise ValueError(f'{name} has only one class. Need more diverse data or another split seed')
    if mode == 'domain':
        groups = [set(df.iloc[ix].domain) for ix in splits.values()]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return splits


def fit_vocab(urls):
    # 0=PAD, 1=UNK. Preserve path/query letter case.
    return {c: i + 2 for i, c in enumerate(sorted(set(''.join(urls))))}


def encode_urls(urls, vocab, max_len):
    x = np.zeros((len(urls), max_len), dtype=np.int32)
    for i, u in enumerate(urls):
        # Keep the prefix; report truncation rate and test max_len as an ablation.
        tokens = [vocab.get(c, 1) for c in str(u)[:max_len]]
        x[i, :len(tokens)] = tokens
    return x


def choose_threshold(y, p):
    # Validation only. Deterministic ties prefer thresholds closest to 0.5.
    candidates = np.linspace(.05, .95, 181)
    return float(max(candidates, key=lambda t: (f1_score(y, p >= t, zero_division=0), -abs(t - .5))))


def evaluate(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'accuracy': float(accuracy_score(y, pred)),
            'precision_phishing': float(precision_score(y, pred, zero_division=0)),
            'recall_phishing': float(recall_score(y, pred, zero_division=0)),
            'f1_phishing': float(f1_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'false_positive_rate': float(fp / (fp + tn)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
