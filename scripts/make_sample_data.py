"""Generate the small placeholder corpus in sample_data/ used by tests and the
offline smoke run. These are NOT PRV201 materials; they are short notes we wrote
so the pipeline can be exercised before/without the real course files."""
import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "sample_data"
OUT.mkdir(exist_ok=True)

pages = [
    ("k-Nearest Neighbors (k-NN)",
     ["k-NN is a non-parametric, instance-based classifier.",
      "To classify a new point, find the k training points closest to it",
      "(usually by Euclidean distance) and take a majority vote of their labels.",
      "There is no training phase beyond storing the data; the cost is paid at prediction time."]),
    ("Choosing k",
     ["Small k (e.g. k = 1) gives a flexible boundary: low bias, high variance, prone to overfitting.",
      "Large k smooths the boundary: higher bias, lower variance, can underfit.",
      "Pick k with cross-validation; odd k avoids ties in binary problems.",
      "Features must be scaled (e.g. StandardScaler) because distance is scale-sensitive."]),
    ("k-NN in scikit-learn",
     ["from sklearn.neighbors import KNeighborsClassifier",
      "model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)",
      "model.score(X_test, y_test) returns test accuracy.",
      "See the Lecture 6 lab notebook for the full iris example."]),
]
c = canvas.Canvas(str(OUT / "L06_knn_sample.pdf"), pagesize=letter)
for title, lines in pages:
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 720, title)
    c.setFont("Helvetica", 12)
    for i, line in enumerate(lines):
        c.drawString(72, 680 - 22 * i, line)
    c.showPage()
c.save()

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
    {"cell_type": "markdown", "metadata": {}, "source": "# Lecture 6 lab: k-NN on the iris dataset\nWe train a k-NN classifier and measure test accuracy for different k."},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source":
     "from sklearn.datasets import load_iris\nfrom sklearn.model_selection import train_test_split\n"
     "X, y = load_iris(return_X_y=True)\n"
     "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)"},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source":
     "from sklearn.neighbors import KNeighborsClassifier\nfor k in [1, 5, 15]:\n"
     "    acc = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)\n"
     "    print(k, round(acc, 3))"},
]}
(OUT / "L06_knn_lab_sample.ipynb").write_text(json.dumps(nb, indent=1))

(OUT / "L03_evaluation_sample.md").write_text(
    "# Lecture 3: Model evaluation\n\n"
    "We hold out a test set so we can estimate how the model performs on data it has never seen. "
    "Evaluating on the training data rewards memorization and hides overfitting. "
    "Cross-validation splits the training data into k folds, trains on k-1 folds and validates on the "
    "remaining fold, rotating so every fold is used once for validation; the average score is a more "
    "stable estimate than a single split. Accuracy can mislead on imbalanced classes, so we also report "
    "precision, recall and the F1 score.\n")
print("wrote", sorted(p.name for p in OUT.iterdir()))
