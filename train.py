"""Train TN1-TN4 on one shared split. Test is never used for model selection."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from common import load_data, split_data, fit_vocab, encode_urls, save_json, choose_threshold, evaluate
from features import feature_matrix
from model import build_model, get_extractor


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='runs/domain_seed42')
    ap.add_argument('--split', choices=['domain', 'random'], default='domain')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--embedding-dim', type=int, default=64)
    ap.add_argument('--filters', type=int, default=64)
    ap.add_argument('--kernel-size', type=int, default=5)
    ap.add_argument('--lstm-units', type=int, default=64)
    ap.add_argument('--deep-dim', type=int, default=64)
    ap.add_argument('--trees', type=int, default=300)
    ap.add_argument('--rf-max-depth', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=2)
    args = ap.parse_args()
    if min(args.epochs, args.batch_size, args.max_len, args.trees, args.deep_dim) < 1:
        ap.error('epochs, batch size, length, trees and dimensions must be positive')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty. Use a new --out to preserve the previous experiment.')
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print('GPU:', tf.config.list_physical_devices('GPU'), flush=True)
    df, cleaning = load_data(args.data)
    splits = split_data(df, args.split, args.seed)
    save_json(out / 'cleaning_report.json', cleaning)
    manifest = df[['source_row', 'url', 'label', 'domain']].copy()
    manifest['split'] = ''
    for name, ix in splits.items():
        manifest.loc[ix, 'split'] = name
    manifest.to_csv(out / 'split_manifest.csv', index=False)
    frames = {k: df.iloc[ix].reset_index(drop=True) for k, ix in splits.items()}
    y = {k: f.label.to_numpy(dtype=np.int32) for k, f in frames.items()}
    vocab = fit_vocab(frames['train'].url.tolist())
    save_json(out / 'vocab.json', vocab)
    cfg = vars(args).copy()
    cfg['label_mapping'] = {'0': 'legitimate', '1': 'phishing'}
    cfg['data_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    cfg['versions'] = {p: importlib.metadata.version(p) for p in
                       ['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'tldextract', 'joblib']}
    save_json(out / 'config.json', cfg)
    X = {k: encode_urls(f.url.tolist(), vocab, args.max_len) for k, f in frames.items()}
    raw_hand = {}
    start = time.perf_counter()
    for k, f in frames.items():
        raw_hand[k], hand_names = feature_matrix(f.url.tolist())
    # Tree models do not need StandardScaler; preserve the feature units.
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    hand = {'train': imputer.fit_transform(raw_hand['train'])}
    hand.update({k: imputer.transform(raw_hand[k]) for k in ['val', 'test']})
    joblib.dump(imputer, out / 'hand_imputer.joblib')
    transformed_names = list(imputer.get_feature_names_out(hand_names))
    save_json(out / 'feature_schema.json', {'raw': hand_names, 'after_imputation': transformed_names})
    prep_seconds = time.perf_counter() - start
    save_json(out / 'data_summary.json', {
        k: {'rows': len(f), 'domains': int(f.domain.nunique()),
            'phishing_fraction': float(f.label.mean()),
            'truncation_fraction': float((f.url.str.len() > args.max_len).mean()),
            'missing_fraction_by_feature': dict(zip(hand_names, np.isnan(raw_hand[k]).mean(axis=0).tolist()))}
        for k, f in frames.items()})
    model = build_model(len(vocab) + 2, args.max_len, args.embedding_dim, args.filters,
                        args.kernel_size, args.lstm_units, args.deep_dim)
    with (out / 'model_summary.txt').open('w', encoding='utf-8') as fh:
        model.summary(print_fn=lambda line: fh.write(line + '\n'))
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y['train'])
    start = time.perf_counter()
    history = model.fit(X['train'], y['train'], validation_data=(X['val'], y['val']),
                        epochs=args.epochs, batch_size=args.batch_size,
                        class_weight={0: float(weights[0]), 1: float(weights[1])},
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                                    restore_best_weights=True)], verbose=1)
    deep_train_seconds = time.perf_counter() - start
    model.save(out / 'cnn_lstm.keras')
    pd.DataFrame(history.history).to_csv(out / 'history.csv', index=False)
    extractor = get_extractor(model)
    start = time.perf_counter()
    deep = {k: extractor.predict(X[k], batch_size=args.batch_size, verbose=0) for k in frames}
    deep_extract_seconds = time.perf_counter() - start
    combined = {k: np.concatenate([deep[k], hand[k]], axis=1) for k in frames}
    predictions = {'TN2_CNN_LSTM': {k: model.predict(X[k], batch_size=args.batch_size, verbose=0).ravel()
                                   for k in ['val', 'test']}}
    timing = {'hand_preprocessing_seconds': prep_seconds, 'cnn_lstm_train_seconds': deep_train_seconds,
              'deep_extraction_all_splits_seconds': deep_extract_seconds}
    rf_specs = [('TN1_Hand_RF', hand), ('TN3_Deep_RF', deep), ('TN4_Hybrid_RF', combined)]
    hybrid_rf = None
    for name, data in rf_specs:
        print('Training', name, 'input shape:', data['train'].shape, flush=True)
        rf = RandomForestClassifier(n_estimators=args.trees, max_depth=args.rf_max_depth,
                                    min_samples_leaf=2, max_features='sqrt',
                                    class_weight='balanced_subsample', n_jobs=args.jobs, random_state=args.seed)
        start = time.perf_counter()
        rf.fit(data['train'], y['train'])
        timing[name + '_rf_fit_seconds'] = time.perf_counter() - start
        joblib.dump(rf, out / (name + '.joblib'), compress=3)
        predictions[name] = {k: rf.predict_proba(data[k])[:, 1] for k in ['val', 'test']}
        if name == 'TN4_Hybrid_RF':
            hybrid_rf = rf
            names = [f'deep_{i}' for i in range(args.deep_dim)] + transformed_names
            pd.DataFrame({'feature': names, 'mdi_importance': rf.feature_importances_}).sort_values(
                'mdi_importance', ascending=False).to_csv(out / 'hybrid_feature_importance.csv', index=False)
    # Lock every threshold from validation before inspecting final test metrics.
    thresholds = {name: choose_threshold(y['val'], p['val']) for name, p in predictions.items()}
    save_json(out / 'thresholds.json', thresholds)
    rows = []
    prediction_table = frames['test'][['url', 'label', 'domain']].copy()
    for name, p in predictions.items():
        for split in ['val', 'test']:
            for policy, threshold in [('fixed_0.5', .5), ('validation_f1', thresholds[name])]:
                rows.append({'model': name, 'split': split, 'threshold_policy': policy,
                             **evaluate(y[split], p[split], threshold)})
        prediction_table[name + '_prob'] = p['test']
        prediction_table[name + '_pred'] = (p['test'] >= thresholds[name]).astype(int)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    prediction_table.to_csv(out / 'test_predictions.csv', index=False)
    prediction_table[prediction_table.label != prediction_table.TN4_Hybrid_RF_pred].to_csv(
        out / 'hybrid_errors.csv', index=False)
    # Warm model, then benchmark the whole offline prediction path, including lexical extraction.
    sample = frames['test'].url.iloc[:min(128, len(frames['test']))].tolist()
    def infer_batch():
        xx = encode_urls(sample, vocab, args.max_len)
        hh, _ = feature_matrix(sample)
        zz = extractor.predict(xx, verbose=0)
        return hybrid_rf.predict_proba(np.concatenate([zz, imputer.transform(hh)], axis=1))
    infer_batch()
    start = time.perf_counter()
    for _ in range(3):
        infer_batch()
    timing['offline_batch_amortized_ms_per_url'] = (time.perf_counter() - start) * 1000 / (3 * len(sample))
    timing['benchmark_batch_size'] = len(sample)
    timing['benchmark_note'] = 'Includes encoding, lexical features, CNN-LSTM, imputation, RF; excludes disk model loading. Not single-request latency.'
    save_json(out / 'timing.json', timing)
    try:
        from plots import make_plots
        make_plots(out)
    except ImportError:
        print('matplotlib missing: plots skipped')
    print(pd.DataFrame(rows).query("split == 'test' and threshold_policy == 'validation_f1'").to_string(index=False))
    print('Saved experiment:', out.resolve())


if __name__ == '__main__':
    main()
