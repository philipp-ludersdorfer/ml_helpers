import matplotlib.pyplot as plt
from sklearn import metrics


def roc_curve(y_true, y_score, color='mediumseagreen',lw=2):
    """Plots the Receiver Operating Characteristic curve.

    Args:
        y_true - (array) true class labels
        y_score - (array) predicted scores
        color - (str) color of ROC curve
        lw - (int) line width

    Returns a plot of the ROC curve.
    """

    # compute true positive and false positive rates
    fpr, tpr, thr = metrics.roc_curve(y_true=y_true, y_score=y_score)
    # compute area under the roc curve
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)

    # create plot
    plt.figure()
    # draw ROC curve
    plt.plot(fpr, tpr, color=color, lw=lw,
             label='ROC curve (AUC = %0.2f)' % auc)
    # draw reference line
    plt.plot([0, 1], [0, 1], color='black',lw=lw, linestyle='--')
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()


def precision_recall_curve(y_true, y_score, color='purple', lw=2):
    """Plots the Precision-Recall curve.

    Args:
        y_true - (array) true class labels
        y_score - (array) predicted scores
        color - (str) color of ROC curve
        lw - (int) line width

    Returns a plot of the Precision-Recall curve.
    """
    # compute precision and recall
    prec, rec, thr = metrics.precision_recall_curve(y_true=y_true, probas_pred=y_score)
    # compute area under the curve
    auc = metrics.auc(rec, prec)

    # create plot
    plt.figure()
    # Draw precision-recall curve
    plt.plot(rec, prec, color=color, lw=lw, label='PR curve (AUC = %0.2f)' % auc)
    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()


def class_separation(y_true, y_score, bins = 30, alpha = 0.7, colors = ['orangered', 'royalblue']):
    """Plots the distribution of predicted scores per class label.

    Args:
        y_true - (array) true class labels
        y_score - (array) predicted scores
        bins - (int) number of bins
        alpha - (float) transparency
        colors - (list) colors for class labels

    Returns a plot of the score distribution per class.
    """

    # create plot
    plt.figure()
    # plot distribution for all class labels
    for label, idx in enumerate(set(y_true)):
        plt.hist(y_score[y_true == label], bins=bins, alpha=alpha, color=colors[idx], label=str(label))
    plt.grid()
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.title('Class Label Separation')
    plt.legend(loc='upper left')
    plt.show()