# Lecture 3: Model evaluation

We hold out a test set so we can estimate how the model performs on data it has never seen. Evaluating on the training data rewards memorization and hides overfitting. Cross-validation splits the training data into k folds, trains on k-1 folds and validates on the remaining fold, rotating so every fold is used once for validation; the average score is a more stable estimate than a single split. Accuracy can mislead on imbalanced classes, so we also report precision, recall and the F1 score.
