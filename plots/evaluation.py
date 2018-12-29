import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_curve(y_true, y_score, color='dodgerblue',lw=2):
    """Plots the Receiver Operating Characteristic Curve.

    Args:
        y_true - (array) true class labels
        y_score - (array) predicted scores
        color - (str) color of ROC curve
        lw - (int) line width

    Returns a plot of the ROC curve.
    """

    # compute tpr, fpr, and auc
    fpr, tpr, thr = metrics.roc_curve(y_true=y_true, y_score=y_score)
    auc = metrics.roc_auc_score(y_true=y,y_score, y_score=y_score)

    # create plot
    plt.figure()
    # draw ROC curve
    plt.plot(fpr, tpr, color=color, lw=lw,
             label='ROC curve (AUC = %0.2f)' % auc)
    # draw reference line
    plt.plot([0, 1], [0, 1], color='black',lw=lw, linestyle='--')
    # make adjustments
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


def plot_precision_recall_curve():
    """Plots the Receiver Operating Characteristic Curve.

    Args:
        y_true - (array) true class labels
        y_score - (array) predicted scores
        color - (str) color of ROC curve
        lw - (int) line width

    Returns a plot of the ROC curve.
    """

    plt.figure()
    lw = 2
    plt.plot(rec, prec, color='purple',
             lw=lw, label='PR curve (AUC = %0.2f)' % roc_auc)
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precicion-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()