from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def make_plots(out):
    out = Path(out)
    h = pd.read_csv(out / 'history.csv')
    ax = h[['loss', 'val_loss']].plot(xlabel='Epoch (zero based)', ylabel='Binary cross entropy', title='CNN-LSTM training')
    ax.figure.tight_layout()
    ax.figure.savefig(out / 'learning_curve.png', dpi=150)
    plt.close(ax.figure)
    p = pd.read_csv(out / 'test_predictions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(p.label, p.TN4_Hybrid_RF_pred, labels=[0, 1],
                                           display_labels=['legitimate', 'phishing'], ax=axes[0], colorbar=False)
    pr = PrecisionRecallDisplay.from_predictions(p.label, p.TN4_Hybrid_RF_prob, ax=axes[1])
    pr.line_.set_label(f'Classifier (AP = {pr.average_precision})')
    axes[1].legend(loc='lower left', fontsize=8)
    roc = RocCurveDisplay.from_predictions(p.label, p.TN4_Hybrid_RF_prob, ax=axes[2])
    roc.line_.set_label(f'Classifier (AUC = {roc.roc_auc})')
    axes[2].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'hybrid_test_curves.png', dpi=150)
    plt.close(fig)
