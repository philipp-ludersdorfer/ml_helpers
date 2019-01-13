import matplotlib.pyplot as plt
from sklearn import metrics, calibration


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


def calibration_curve(y_true, y_score, calibrate=None, clf=None, x=None, cv=None, bins=10, color='dodgerblue', lw=2):
    """Plots the probability calibration curve.

    The plot shows the fraction of true positives as a function of mean predicted score per score bin.
    The number of examples per score bin (histogram) is plotted on a second vertical axis.

    Args:
        y_true - (array) true class labels
        y_score - (array) predicted scores
        calibrate - (str) 'sigmoid', 'isotomic', or None
        clf - (object) scikit-Learn Classifier object
        cv - (int) number of cross-validation folds for calibration
        bins - (int) number of bins
        color - (list) colors for class labels
        lw - (int) line width
    Returns a plot with the probability calibration curve and a histogram of predicted scores.
    """
    # compute brier score
    brier_score = metrics.brier_score_loss(y_true, y_score, pos_label=y_true.max())
    # compute fraction of true positives and mean predicted score per bin
    prop_true, prop_pred = calibration.calibration_curve(y_true=y_true,
                                                         y_prob=y_score,
                                                         n_bins=bins)

    # calibration
    if calibrate:
        calibrated_clf = calibration.CalibratedClassifierCV(clf, cv=cv, method=calibrate)
        calibrated_clf.fit(x,y_true)
        y_score_cal = calibrated_clf.predict_proba(x)[:,1]
        brier_score_cal = metrics.brier_score_loss(y_true, y_score_cal, pos_label=y_true.max())
        # compute fraction of true positives and mean predicted score per bin
        prop_true_cal, prop_pred_cal = calibration.calibration_curve(y_true=y_true,
                                                                     y_prob=y_score_cal,
                                                                     n_bins=bins)

    # create plot
    plt.figure()
    # draw reliability curve
    plt.plot(prop_pred, prop_true, color=color, lw=lw, label='Uncalibrated (Brier score = %0.2f)' % brier_score)
    # draw calibrated reliability curve
    if calibrate:
        plt.plot(prop_pred_cal, prop_true_cal, color='purple', lw=lw,
                 label='Calibrated (Brier score = %0.2f)' % brier_score_cal)
    # draw reference line
    plt.plot([0, 1], [0, 1], color='black',lw=lw, linestyle='--')
    # label axes
    plt.ylabel('Fraction of positives')
    plt.xlabel('Predicted score')
    plt.legend(loc='lower right')
    # add second vertical axis
    plt.twinx()
    # draw predicted score histogram
    plt.hist(y_score, bins=bins, alpha=0.5, histtype='step', color=color)
    if calibrate:
        plt.hist(y_score_cal, bins=bins, alpha=0.5, histtype='step', color='purple')
    plt.ylabel('Frequency')
    plt.grid()
    plt.title('Calibration Plot (Reliability Curve)')
    plt.show()