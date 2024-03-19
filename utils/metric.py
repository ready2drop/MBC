import os
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Confusion Matrix
def plot_confusion_matrix(conf_matrix, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    else:
        plt.show()

# Plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, class_names=None, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Add class names to the plot
    if class_names:
        for i, txt in enumerate(class_names):
            plt.annotate(txt, (fpr[i], tpr[i]), textcoords="offset points", xytext=(5,5), ha='center')

    if save_path:
        plt.savefig(os.path.join(save_path, "roc_curve.png"))
    else:
        plt.show()
