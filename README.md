# AICS-202 — Adversarial Machine Learning
## Module 1: Baseline Model & Threat Modeling

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![Dataset](https://img.shields.io/badge/Dataset-Sklearn%20Digits-green)
![Status](https://img.shields.io/badge/Module%201-Completed-success)
![Security](https://img.shields.io/badge/Focus-Adversarial%20ML-red)

## 1. Project Overview

This project establishes a baseline machine learning classifier and evaluates its performance under normal, unmodified input conditions as the foundation for subsequent adversarial machine learning experiments.

The practical exercise is part of **ICDFA AICS-202 – Adversarial Machine Learning**. It uses the Scikit-learn Digits dataset and a Logistic Regression classifier to establish baseline predictive performance before investigating adversarial evasion, training-data poisoning, and defensive techniques.

The primary objectives are:

- Load and inspect the handwritten digits dataset.
- Perform a reproducible stratified train-test split.
- Train a baseline Logistic Regression classifier.
- Measure clean-test classification accuracy.
- Generate a confusion matrix.
- Define the security threat model and potential attack surface.
- Establish baseline evidence for subsequent adversarial robustness evaluations.

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| Programming Language | Python 3.10+ |
| Machine Learning Framework | Scikit-learn |
| Numerical Processing | NumPy |
| Data Visualization | Matplotlib |
| Dataset | `sklearn.datasets.load_digits` |
| Model | Logistic Regression |
| Training Configuration | `max_iter=600` |
| Random Seed | `42` |
| Train-Test Strategy | Stratified 75/25 split |
| Execution Environment | Google Colab / Jupyter Notebook |

### Dependencies

```bash
pip install numpy scikit-learn matplotlib notebook
```

## 3. Dataset Description

The experiment uses the built-in Scikit-learn Digits dataset.

Each sample represents a handwritten digit from 0 to 9, represented as an 8×8 grayscale image.

| Dataset Property | Value |
|---|---:|
| Total Samples | 1,797 |
| Input Features | 64 |
| Image Dimensions | 8 × 8 pixels |
| Number of Classes | 10 |
| Training Samples | 1,347 |
| Testing Samples | 450 |
| Original Pixel Range | 0–16 |
| Normalized Pixel Range | 0–1 |

The dataset is divided into training and testing partitions using stratified sampling to preserve the class distribution.

**Dataset separation:**

- Training partition: 75% (1,347 samples).
- Testing partition: 25% (450 samples).

The test partition is reserved for evaluating model generalization and must not be used to train the baseline classifier.

## 4. Baseline Model Architecture

The baseline classifier uses Logistic Regression, a supervised learning algorithm suitable for multiclass classification.

### Model Configuration

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=600,
    random_state=42
)

model.fit(X_train, y_train)
```

### Training Pipeline

1. Load the Digits dataset.
2. Normalize pixel values to the range [0, 1].
3. Split the dataset into training and testing partitions.
4. Initialize the Logistic Regression classifier.
5. Train the classifier using the training partition.
6. Generate predictions using the testing partition.
7. Calculate clean-test accuracy.
8. Generate a confusion matrix for classification analysis.

## 5. Experimental Results

<details>
<summary><strong>Baseline Evaluation Summary</strong></summary>

| Evaluation Metric | Result |
|---|---:|
| Total Dataset Samples | 1,797 |
| Training Samples | 1,347 |
| Testing Samples | 450 |
| Clean Test Accuracy | **96.22%** |
| Correct Predictions | 433 |
| Incorrect Predictions | 17 |

</details>

### Clean-Test Accuracy

**Measured accuracy: 96.22%**

The baseline classifier achieved a clean-test accuracy of 0.9622, indicating that it correctly classified approximately 96.22% of the testing samples under normal, unmodified input conditions.

The result establishes the reference performance for subsequent adversarial machine learning experiments.

### Confusion Matrix

The confusion matrix is used to analyze classification performance across all ten digit classes.

Generate the visualization using:

```python
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

disp.plot(cmap="Blues")
plt.title("Baseline Model - Confusion Matrix")
plt.savefig(
    "results/baseline_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
```

**Evidence:** `results/baseline_confusion_matrix.png`

## 6. Threat Model Analysis

This section defines the security assumptions and potential attack surface of the baseline classifier.

| Threat Model Field | Analysis |
|---|---|
| Protected Model and Asset | The locally trained Logistic Regression classifier, model parameters, training data integrity, and prediction integrity. |
| Legitimate Model Function | Classify handwritten digit images into ten categories (0–9) using 64 numerical pixel features. |
| Potential Attacker-Controlled Input | Digit image inputs that an attacker can manipulate through controlled pixel perturbations within the valid input range. |
| Attacker Knowledge Assumption | White-box. The attacker is assumed to know the model architecture, parameters, preprocessing method, and input representation. |
| Desired Adversarial Outcome | Cause incorrect classification through carefully crafted perturbations while attempting to preserve the original visual meaning of the digit. |
| Monitoring and Prevention Controls | Input validation, adversarial testing, anomaly monitoring, adversarial training, FGSM and PGD evaluation, model version control, and reproducible experiment logging. |

### Threat Model Scope

The practical exercise assumes a local white-box environment in which the attacker has knowledge of the trained model and can manipulate its input features.

The baseline model serves as the target for subsequent adversarial evasion experiments, particularly Fast Gradient Sign Method (FGSM).

## 7. Security Reflection

### Why does good clean-test accuracy not establish security against manipulated inputs?

A clean-test accuracy of 96.22% demonstrates that the Logistic Regression classifier performs well on normal, unmodified testing samples. However, this result does not establish that the classifier is secure against adversarial manipulation.

Adversarial attacks exploit the sensitivity of machine learning models to carefully crafted input modifications. An attacker can introduce small perturbations to pixel values, potentially causing the classifier to produce an incorrect prediction even when the original image is correctly classified.

For example, a handwritten digit may be modified through controlled pixel perturbations so that the classifier predicts a different digit. Although the perturbations may be small, they can significantly influence the model's decision.

Consequently, clean-test accuracy and adversarial robustness measure different aspects of model performance:

- **Clean accuracy:** Measures predictive performance on normal testing data.
- **Adversarial robustness:** Measures the ability to maintain reliable predictions under deliberately manipulated inputs.

A model with high clean-test accuracy may still be vulnerable to adversarial attacks. Therefore, additional evaluations using FGSM, Projected Gradient Descent (PGD), and other robustness-testing techniques are necessary.

## 8. Limitations

The current experiment has several limitations:

1. The evaluation uses a small, built-in handwritten digits dataset.
2. The classifier is evaluated under clean input conditions only.
3. No adversarial training or defensive mechanism has been evaluated in this module.
4. The threat model assumes a white-box attacker.
5. The results cannot be interpreted as evidence of production-level security or robustness.

These limitations define the scope of the baseline experiment and motivate further adversarial security assessments.

## 9. Reproducibility

To reproduce the experiment:

1. Open the baseline notebook in Google Colab or Jupyter Notebook.
2. Install the required dependencies.
3. Execute all notebook cells from beginning to end.
4. Verify the dataset dimensions and train-test partition sizes.
5. Train the Logistic Regression classifier.
6. Confirm the clean-test accuracy.
7. Generate and save the confusion
