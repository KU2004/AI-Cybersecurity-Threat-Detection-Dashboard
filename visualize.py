import matplotlib
matplotlib.use('TkAgg')  # 🔥 Force GUI

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

def plot_confusion_matrix(y_true, y_pred):
    os.makedirs("images", exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")

    plt.savefig("images/confusion_matrix.png")

    # 🔥 THIS OPENS NEW WINDOW
    plt.show(block=True)