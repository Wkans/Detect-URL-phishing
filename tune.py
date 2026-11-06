"""Hyperparameter search for CNN-LSTM + Random Forest (TN4) with Optuna.

Selection uses the validation split only; the test split is never encoded, scored or inspected.
Stage 1 tunes the CNN-LSTM (each trial: train network -> extract deep vector -> fit a fixed RF -> val F1).
Stage 2 freezes the best network and tunes only the Random Forest on cached features (cheap).
Afterwards run the printed train.py command to produce the final experiment and test metrics.
"""
import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from common import load_data, split_data, fit_vocab, encode_urls, save_json, choose_threshold
from features import feature_matrix
from model import build_model, get_extractor

try:
    import optuna
except ImportError:
    raise SystemExit('optuna is required: pip install "optuna>=3.6,<5"')


def val_f1(y, p):
    return float(f1_score(y, p >= choose_threshold(y, p), zero_division=0))


class PruneCallback(tf.keras.callbacks.Callback):
    """Report val_loss each epoch so Optuna can stop clearly bad trials early."""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(float(logs['val_loss']), epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='runs/tune_seed42')
    ap.add_argument('--split', choices=['domain', 'random'], default='domain')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--max-per-domain', type=int, default=0, help='Randomly keep at most N rows per domain (0 = no cap)')
    ap.add_argument('--max-train', type=int, default=0, help='Subsample N training rows per trial for speed (0 = all)')
    ap.add_argument('--trials', type=int, default=20, help='CNN-LSTM trials (stage 1)')
    ap.add_argument('--rf-trials', type=int, default=40, help='Random Forest trials (stage 2)')
    ap.add_argument('--epochs', type=int, default=15, help='Max epochs per trial')
    ap.add_argument('--patience', type=int, default=5, help='EarlyStopping patience (epochs without val_loss improvement)')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--timeout', type=int, default=0, help='Stop stage 1 after N seconds (0 = no limit)')
    ap.add_argument('--resume', action='store_true', help='Continue a previous search stored in --out/tune.db')
    args = ap.parse_args()
    if min(args.max_len, args.trials, args.rf_trials, args.epochs, args.patience) < 1:
        ap.error('max-len, trials, rf-trials, epochs and patience must be positive')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty. Use a new --out, or --resume to continue it.')
    out.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(out / 'tune.db').resolve().as_posix()}"  # every trial is saved as soon as it ends
    tf.keras.utils.set_random_seed(args.seed)

    df, _ = load_data(args.data)
    if args.max_per_domain:
        shuffled = df.sample(frac=1, random_state=args.seed)
        df = shuffled[shuffled.groupby('domain').cumcount() < args.max_per_domain].sort_index().reset_index(drop=True)
    splits = split_data(df, args.split, args.seed)
    frames = {k: df.iloc[splits[k]].reset_index(drop=True) for k in ['train', 'val']}  # test deliberately not loaded
    y = {k: f.label.to_numpy(dtype=np.int32) for k, f in frames.items()}
    vocab = fit_vocab(frames['train'].url.tolist())
    X = {k: encode_urls(f.url.tolist(), vocab, args.max_len) for k, f in frames.items()}
    raw = {k: feature_matrix(f.url.tolist())[0] for k, f in frames.items()}
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    hand = {'train': imputer.fit_transform(raw['train']), 'val': imputer.transform(raw['val'])}
    print({k: len(f) for k, f in frames.items()}, 'hand features:', hand['train'].shape[1], flush=True)

    # Optional fixed subsample of the training set so each trial is faster; val stays complete.
    rng = np.random.RandomState(args.seed)
    tr_ix = np.arange(len(y['train']))
    if args.max_train and args.max_train < len(tr_ix):
        tr_ix = np.sort(rng.choice(tr_ix, args.max_train, replace=False))
        if len(np.unique(y['train'][tr_ix])) < 2:
            raise ValueError('Subsample has one class; increase --max-train')
    Xtr, ytr, htr = X['train'][tr_ix], y['train'][tr_ix], hand['train'][tr_ix]
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr)
    cw = {0: float(weights[0]), 1: float(weights[1])}

    def make_rf(params, n_jobs):
        return RandomForestClassifier(class_weight='balanced_subsample', n_jobs=n_jobs,
                                      random_state=args.seed, **params)

    # ---------------- Stage 1: CNN-LSTM ----------------
    base_rf = dict(n_estimators=150, max_depth=20, min_samples_leaf=2, max_features='sqrt')

    def objective_net(trial):
        p = {'embedding_dim': trial.suggest_categorical('embedding_dim', [32, 64, 128]),
             'filters': trial.suggest_categorical('filters', [32, 64, 128]),
             'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
             'lstm_units': trial.suggest_categorical('lstm_units', [32, 64, 128]),
             'deep_dim': trial.suggest_categorical('deep_dim', [32, 64, 128]),
             'learning_rate': trial.suggest_float('learning_rate', 3e-4, 3e-3, log=True),
             'dropout': trial.suggest_float('dropout', .1, .5),
             'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])}
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        batch = p.pop('batch_size')
        model = build_model(len(vocab) + 2, args.max_len, **p)
        try:
            model.fit(Xtr, ytr, validation_data=(X['val'], y['val']), epochs=args.epochs, batch_size=batch,
                      class_weight=cw, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                                                  restore_best_weights=True),
                                 PruneCallback(trial)])
            ext = get_extractor(model)
            f = {'train': np.concatenate([ext.predict(Xtr, batch_size=256, verbose=0), htr], axis=1),
                 'val': np.concatenate([ext.predict(X['val'], batch_size=256, verbose=0), hand['val']], axis=1)}
            rf = make_rf(base_rf, args.jobs).fit(f['train'], ytr)
            score = val_f1(y['val'], rf.predict_proba(f['val'])[:, 1])
        finally:
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        trial.set_user_attr('batch_size', batch)
        return score

    study_net = optuna.create_study(direction='maximize', study_name='cnn_lstm', storage=storage, load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=args.seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))

    def save_net_csv(study, trial):
        study.trials_dataframe().to_csv(out / 'trials_cnn_lstm.csv', index=False)

    finished = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
                   for t in study_net.trials)
    if finished:
        print(f'Resuming CNN-LSTM search: {finished} trials already finished', flush=True)
    start = time.perf_counter()
    if args.trials > finished:
        study_net.optimize(objective_net, n_trials=args.trials - finished, timeout=args.timeout or None,
                           gc_after_trial=True, callbacks=[save_net_csv])
    stage1_seconds = time.perf_counter() - start
    study_net.trials_dataframe().to_csv(out / 'trials_cnn_lstm.csv', index=False)
    done = [t for t in study_net.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit('No CNN-LSTM trial completed; increase --trials/--timeout or reduce --max-train')
    best_net = dict(study_net.best_trial.params)
    print('Best CNN-LSTM:', best_net, 'val F1 =', round(study_net.best_value, 5), flush=True)

    # ---------------- Stage 2: Random Forest on the best network ----------------
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    net = {k: v for k, v in best_net.items() if k != 'batch_size'}
    batch = best_net['batch_size']
    model = build_model(len(vocab) + 2, args.max_len, **net)
    model.fit(Xtr, ytr, validation_data=(X['val'], y['val']), epochs=args.epochs, batch_size=batch,
              class_weight=cw, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)])
    ext = get_extractor(model)
    feats = {'train': np.concatenate([ext.predict(Xtr, batch_size=256, verbose=0), htr], axis=1),
             'val': np.concatenate([ext.predict(X['val'], batch_size=256, verbose=0), hand['val']], axis=1)}

    def objective_rf(trial):
        p = {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
             'max_depth': trial.suggest_int('max_depth', 8, 40, step=4),
             'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
             'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5])}
        rf = make_rf(p, args.jobs).fit(feats['train'], ytr)
        return val_f1(y['val'], rf.predict_proba(feats['val'])[:, 1])

    study_rf = optuna.create_study(direction='maximize', study_name='random_forest', storage=storage,
                                   load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed))

    def save_rf_csv(study, trial):
        study.trials_dataframe().to_csv(out / 'trials_random_forest.csv', index=False)

    rf_done = len([t for t in study_rf.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if args.rf_trials > rf_done:
        study_rf.optimize(objective_rf, n_trials=args.rf_trials - rf_done, gc_after_trial=True,
                          callbacks=[save_rf_csv])
    study_rf.trials_dataframe().to_csv(out / 'trials_random_forest.csv', index=False)
    best_rf = dict(study_rf.best_trial.params)
    print('Best Random Forest:', best_rf, 'val F1 =', round(study_rf.best_value, 5), flush=True)

    result = {'objective': 'validation F1 (threshold chosen on validation)', 'seed': args.seed,
              'split': args.split, 'max_len': args.max_len, 'max_train': args.max_train,
              'epochs_per_trial': args.epochs, 'patience': args.patience, 'cnn_lstm': best_net, 'cnn_lstm_val_f1': study_net.best_value,
              'random_forest': best_rf, 'random_forest_val_f1': study_rf.best_value,
              'trials_completed': len(done), 'trials_pruned': sum(t.state == optuna.trial.TrialState.PRUNED
                                                                    for t in study_net.trials),
              'stage1_seconds': stage1_seconds}
    save_json(out / 'best_params.json', result)
    cmd = (f"python train.py --data {args.data} --split {args.split} --seed {args.seed} --max-len {args.max_len} "
           f"--embedding-dim {best_net['embedding_dim']} --filters {best_net['filters']} "
           f"--kernel-size {best_net['kernel_size']} --lstm-units {best_net['lstm_units']} "
           f"--deep-dim {best_net['deep_dim']} --batch-size {best_net['batch_size']} "
           f"--learning-rate {best_net['learning_rate']:.6g} --dropout {best_net['dropout']:.4f} "
           f"--trees {best_rf['n_estimators']} --rf-max-depth {best_rf['max_depth']} "
           f"--rf-min-leaf {best_rf['min_samples_leaf']} --rf-max-features {best_rf['max_features']} "
           f"--out runs/tuned_seed{args.seed}")
    (out / 'train_command.txt').write_text(cmd + '\n', encoding='utf-8')
    print('\nSaved search:', out.resolve())
    print('Next:', cmd)


if __name__ == '__main__':
    main()
