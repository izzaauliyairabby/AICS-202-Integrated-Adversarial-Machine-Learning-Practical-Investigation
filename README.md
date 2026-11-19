# AICS-202 — Adversarial Machine Learning
## Comprehensive Laboratory: Baseline, FGSM Evasion, Data Poisoning & Robustness Defences

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Security](https://img.shields.io/badge/Domain-Adversarial%20ML-red)
![Laboratories](https://img.shields.io/badge/Laboratories-01--04-success)

**Course:** ICDFA AICS-202 – Adversarial Machine Learning  
**Laboratories:** 01, 02, 03, and 04  
**Execution Environment:** Google Colab / Jupyter Notebook  
**Dataset:** Scikit-learn Digits Dataset  
**Baseline Algorithm:** Logistic Regression  
**Attack Techniques:** FGSM and PGD  
**Security Scope:** Controlled Local Machine Learning Environment

---

## 1. Project Overview

This repository contains four practical laboratory exercises exploring adversarial machine learning security, from baseline model development to adversarial attacks, training-data poisoning, and defensive robustness assessment.

The experiments use a locally trained Logistic Regression classifier and the built-in Scikit-learn Digits dataset.

The primary objective is to understand how machine learning systems behave under normal operating conditions and how their predictive performance and data integrity can be affected by adversarial manipulation.

The project is organized into four laboratory modules:

| Laboratory | Module | Main Focus |
|---|---|---|
| Lab 01 | Baseline Model & Threat Modelling | Dataset preparation, baseline training, clean evaluation, and threat modeling |
| Lab 02 | FGSM Evasion Testing | Gradient-based adversarial example generation and evasion evaluation |
| Lab 03 | Training Data Poisoning & Dataset Integrity | Training-label corruption and data integrity controls |
| Lab 04 | Defensive Retraining & Robustness Assessment | Adversarial augmentation, FGSM, PGD, and defensive evaluation |

All experiments are conducted in a controlled local environment. No production AI systems, external services, or real user accounts are targeted.

## 2. Project Objectives

- Establish a reproducible machine learning baseline.
- Evaluate clean-test classification performance.
- Identify machine learning assets, attack surfaces, and adversary capabilities.
- Understand white-box adversarial attack assumptions.
- Implement Fast Gradient Sign Method (FGSM).
- Evaluate model accuracy under different perturbation magnitudes.
- Simulate training-data integrity compromise through random label corruption.
- Measure the effect of corrupted training labels on model performance.
- Identify technical and procedural data integrity safeguards.
- Implement simple adversarial augmentation.
- Evaluate baseline and augmented models using FGSM and PGD.
- Analyze the limitations of defensive retraining.
- Develop evidence-based robustness recommendations.

## 3. Experimental Environment

| Component | Configuration |
|---|---|
| Programming Language | Python 3.10+ |
| Machine Learning Framework | Scikit-learn |
| Numerical Processing | NumPy |
| Data Visualization | Matplotlib |
| Dataset | `sklearn.datasets.load_digits()` |
| Classification Algorithm | Logistic Regression |
| Maximum Iterations | 600 |
| Baseline Random Seed | 42 |
| Train-Test Split | 75% / 25% |
| Evasion Attack | FGSM |
| Iterative Attack | PGD |
| Poisoning Simulation | Random Training-Label Corruption |
| Defensive Technique | FGSM Adversarial Augmentation |
| Execution Environment | Google Colab / Jupyter Notebook |

### Dependencies

```bash
pip install numpy matplotlib scikit-learn notebook
```

## 4. Dataset Description

The experiments use the built-in Scikit-learn Digits dataset.

Each sample represents a handwritten digit from 0 to 9, encoded as an 8 × 8 grayscale image containing 64 numerical pixel features.

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

### Data Preprocessing

Pixel values are normalized to the range [0, 1].

```python
digits = load_digits()

X = digits.data.astype(float) / 16.0
y = digits.target.astype(int)
```

The dataset is divided using stratified sampling:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

The training partition contains 1,347 samples, while the testing partition contains 450 samples.

The testing partition remains unchanged during the poisoning experiment.

---

# 5. Laboratory 01 — Baseline Model & Threat Modelling

## 5.1 Objective

Laboratory 01 establishes the baseline classification model and defines the security assumptions for subsequent adversarial experiments.

The main activities include:

- Loading the local Digits dataset.
- Preparing the training and testing partitions.
- Training a Logistic Regression classifier.
- Measuring clean-test accuracy.
- Generating a confusion matrix.
- Defining the threat model and attack surface.

## 5.2 Baseline Model Configuration

The baseline classifier uses Logistic Regression.

```python
model = LogisticRegression(
    max_iter=600,
    random_state=42
)

model.fit(X_train, y_train)
```

## 5.3 Baseline Results

<details>
<summary><strong>Baseline Evaluation Results</strong></summary>

| Metric | Result |
|---|---:|
| Total Samples | 1,797 |
| Training Samples | 1,347 |
| Testing Samples | 450 |
| Clean-Test Accuracy | **96.22%** |

</details>

The model achieved a clean-test accuracy of 0.9622, demonstrating strong classification performance on unmodified test samples.

This result establishes the reference benchmark for subsequent adversarial experiments.

## 5.4 Confusion Matrix

The confusion matrix evaluates the classifier's predictions across all ten digit classes.

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Baseline Confusion Matrix")
plt.show()
```

## 5.5 Threat Model

| Threat Model Component | Description |
|---|---|
| Protected Model and Asset | Locally trained Logistic Regression classifier, model parameters, training data integrity, and prediction integrity |
| Legitimate Model Function | Classify handwritten digit images into ten categories |
| Potential Attacker-Controlled Input | Normalized 64-feature digit images |
| Attacker Knowledge | White-box access to model architecture, parameters, and preprocessing |
| Desired Adversarial Outcome | Cause incorrect classification through controlled input manipulation |
| Monitoring and Prevention Controls | Input validation, adversarial testing, anomaly monitoring, access controls, model versioning, and reproducible experiment logging |

## 5.6 Security Reflection

High clean-test accuracy does not establish security against manipulated inputs.

The baseline accuracy of 96.22% measures the model's ability to classify normal, unmodified samples. It does not measure how well the classifier responds to deliberately crafted adversarial inputs.

A small modification to pixel features may change the model's prediction, even when the original image is correctly classified.

Therefore, predictive accuracy and adversarial robustness must be evaluated separately.

---

# 6. Laboratory 02 — FGSM Evasion Testing

## 6.1 Objective

Laboratory 02 investigates how controlled adversarial perturbations affect the baseline classifier.

The experiment implements the Fast Gradient Sign Method (FGSM), a gradient-based attack that modifies input features according to the sign of the loss gradient.

The practical assumes a white-box attacker with access to the locally trained model and its input gradients.

## 6.2 Input Gradient Calculation

The notebook calculates the cross-entropy gradient with respect to the normalized input features.

```python
def input_loss_gradient(model, features, labels):
    probs = model.predict_proba(features)

    targets = np.zeros_like(probs)
    label_indices = np.searchsorted(
        model.classes_,
        labels
    )

    targets[
        np.arange(len(labels)),
        label_indices
    ] = 1.0

    return (probs - targets) @ model.coef_
```

## 6.3 FGSM Implementation

```python
def fgsm(model, features, labels, epsilon=0.10):
    grad = input_loss_gradient(
        model,
        features,
        labels
    )

    return np.clip(
        features + epsilon * np.sign(grad),
        0.0,
        1.0
    )
```

The clipping operation ensures that the manipulated pixel features remain within the valid normalized input range of [0, 1].

## 6.4 Epsilon Evaluation

The notebook evaluates the classifier using the following perturbation magnitudes:

```python
epsilons = [0.00, 0.03, 0.07, 0.12, 0.18]
```

### Experimental Results

| Epsilon | Accuracy Under FGSM |
|---:|---:|
| 0.00 | 96.2% |
| 0.03 | 90.7% |
| 0.07 | 79.1% |
| 0.12 | 51.6% |
| 0.18 | 20.4% |

<details>
<summary><strong>FGSM Results Analysis</strong></summary>

The experiment demonstrates a substantial reduction in classification accuracy as perturbation strength increases.

Key observations:

- At epsilon 0.00, the classifier maintains its original clean-test accuracy.
- At epsilon 0.03, accuracy decreases to 90.7%.
- At epsilon 0.07, accuracy decreases to 79.1%.
- At epsilon 0.12, accuracy decreases to 51.6%.
- At epsilon 0.18, accuracy decreases to 20.4%.

These results demonstrate that the baseline classifier is sensitive to gradient-based input perturbations under the tested white-box threat model.

</details>

## 6.5 Adversarial Example Visualization

The notebook compares clean and adversarial samples using multiple perturbation values:

| Visualization Experiment | Epsilon |
|---|---:|
| Experiment 1 | 0.01 |
| Experiment 2 | 0.05 |
| Experiment 3 | 0.10 |
| Experiment 4 | 0.15 |
| Experiment 5 | 0.20 |

The notebook displays original and modified images alongside their model predictions.

It also measures the maximum observed L-infinity perturbation:

```python
np.max(np.abs(adversarial - X_test))
```

The maximum observed changes correspond to the selected epsilon values, subject to floating-point precision and input clipping.

## 6.6 Additional Student Evaluation

The laboratory requires an additional student-selected epsilon between 0.01 and 0.20.

The additional experiment should record:

- Selected epsilon.
- Clean-test accuracy.
- Attacked-test accuracy.
- Maximum observed perturbation.
- An example prediction change.

This additional experiment is separate from the five epsilon values in the existing robustness curve.

---

# 7. Laboratory 03 — Training Data Poisoning & Dataset Integrity

## 7.1 Objective

Laboratory 03 evaluates the impact of deliberately corrupted training labels on supervised machine learning performance.

The experiment simulates a training-data integrity compromise by randomly changing labels in the training partition.

**Important:** This is an introductory random label-corruption simulation, not a realistic targeted poisoning or backdoor attack.

The testing partition and its labels remain unchanged.

## 7.2 Poisoning Simulation Methodology

The notebook evaluates multiple training-label corruption levels:

```python
corruption_levels = [
    0.00, 0.05, 0.10, 0.20,
    0.30, 0.40, 0.50, 0.60,
    0.70, 0.80, 0.90, 1.00
]
```

At each corruption level:

1. Copy the original training labels.
2. Select a random subset of training samples.
3. Replace selected labels with different digit classes.
4. Train a new Logistic Regression classifier.
5. Evaluate the resulting model on the unchanged test partition.
6. Record the clean-test accuracy.

The experiment uses a NumPy random generator with seed `2026` for the corruption simulation.

## 7.3 Experimental Results

<badge color="success">Recorded notebook results</badge>

| Training-Label Corruption | Changed Labels | Clean-Test Accuracy |
|---:|---:|---:|
| 0% | 0 | 96.2% |
| 5% | 67 | 96.0% |
| 10% | 135 | 95.8% |
| 20% | 269 | 93.3% |
| 30% | 404 | 92.4% |
| 40% | 539 | 89.3% |
| 50% | 674 | 86.0% |
| 60% | 808 | 84.0% |
| 70% | 943 | 67.6% |
| 80% | 1,078 | 54.2% |
| 90% | 1,212 | 8.7% |
| 100% | 1,347 | 0.7% |

## 7.4 Results Analysis

The experiment demonstrates that training-label corruption can significantly reduce the classification performance of a supervised learning model.

Key observations:

- Low corruption levels produce relatively small performance changes.
- Increasing label corruption generally reduces clean-test accuracy.
- At 50% corruption, accuracy decreases to 86.0%.
- At 70% corruption, accuracy decreases to 67.6%.
- At 90% corruption, accuracy falls to 8.7%.
- At 100% corruption, accuracy reaches approximately 0.7%.

The results show that training-data integrity is a critical security asset.

The small non-monotonic changes between some corruption levels can occur because the experiment uses random label selection and replacement. Different randomly selected samples may have different effects on model training.

## 7.5 Dataset Integrity Controls

The laboratory requires a dataset integrity checklist covering the following control categories.

| Control Category | Recommended Safeguard |
|---|---|
| Dataset Provenance | Maintain documented dataset origin, ownership, and acquisition records |
| Cryptographic Hashes | Generate SHA-256 hashes to detect unauthorized dataset modification |
| Label Audits | Review label distributions and investigate unexpected changes |
| Role-Restricted Writes | Restrict dataset modification privileges to authorized personnel |
| Staged Approval | Require review and approval before promoting training datasets |
| Anomaly Review | Monitor unusual class distributions and suspicious label changes |

### Additional Safeguard Recommendations

**Technical safeguards:**

1. Use cryptographic checksums to verify dataset integrity.
2. Implement automated label-distribution validation and anomaly detection.
3. Maintain immutable dataset versions and access-controlled storage.

**Procedural safeguards:**

1. Require independent approval before accepting modified training datasets.
2. Maintain audit logs documenting dataset access, changes, and approvals.

These controls are recommendations for the hypothetical dataset workflow; they are not all implemented as automated controls in the notebook.

## 7.6 Security Reflection

Training data is a security asset because supervised machine learning models depend on the quality and integrity of their training labels.

Unauthorized modifications can influence the learned decision boundaries and reduce prediction reliability.

Even when the model architecture and training code remain unchanged, compromised labels can undermine the model's performance.

Therefore, machine learning security must protect not only model parameters and prediction interfaces but also the data supply chain.

---

# 8. Laboratory 04 — Defensive Retraining & Robustness Assessment

## 8.1 Objective

Laboratory 04 compares a baseline classifier with a model trained using simple adversarial augmentation.

Both models are evaluated under clean conditions and adversarial attacks.

The notebook uses:

- Clean-test evaluation.
- FGSM adversarial evaluation.
- Projected Gradient Descent (PGD) evaluation.

An important principle of this laboratory is that adversarial augmentation does not guarantee improved robustness. Attacks must be evaluated against the defended model itself.

## 8.2 Adversarial Training Configuration

The baseline model is trained using the original training dataset.

Adversarial samples are then generated from the training partition using FGSM with epsilon 0.10.

```python
train_adversarial = fgsm(
    model,
    X_train,
    y_train,
    epsilon=0.10
)

aug_features = np.concatenate([
    X_train,
    train_adversarial
])

aug_labels = np.concatenate([
    y_train,
    y_train
])

defended = build_model(
    aug_features,
    aug_labels
)
```

The augmented dataset combines original and adversarial training samples while retaining the original labels.

This produces a training dataset containing both clean and adversarial examples.

## 8.3 PGD Attack Configuration

The notebook implements a small iterative PGD variant.

```python
def pgd(
    model,
    features,
    labels,
    epsilon=0.12,
    alpha=0.03,
    steps=7,
    seed=42
):
    rng = np.random.default_rng(seed)

    original = features.copy()

    perturbed = np.clip(
        original + rng.uniform(
            -epsilon,
            epsilon,
            size=original.shape
        ),
        0,
        1
    )

    for _ in range(steps):
        grad = input_loss_gradient(
            model,
            perturbed,
            labels
        )

        perturbed = (
            perturbed + alpha * np.sign(grad)
        )

        perturbed = np.maximum(
            np.minimum(
                perturbed,
                original + epsilon
            ),
            original - epsilon
        )

        perturbed = np.clip(
            perturbed,
            0.0,
            1.0
        )

    return perturbed
```

### PGD Parameters

| Parameter | Value |
|---|---:|
| Epsilon | 0.12 |
| Step Size (Alpha) | 0.03 |
| Iterations | 7 |
| Random Seed | 42 |
| Input Constraint | [0, 1] |

## 8.4 Defensive Evaluation Methodology

Both models are evaluated using the same test partition.

| Evaluation Type | Configuration |
|---|---|
| Clean | Original, unmodified test samples |
| FGSM | Epsilon 0.12 |
| PGD | Epsilon 0.12, alpha 0.03, 7 steps |

The attacks are generated using each evaluated model's own input gradients.

This allows the experiment to evaluate the defended model under attacks adapted to its parameters.

## 8.5 Experimental Results

<badge color="warning">Defensive performance did not improve</badge>

| Model | Clean Accuracy | FGSM Accuracy | PGD Accuracy |
|---|---:|---:|---:|
| Baseline | 96.2% | 51.6% | 48.9% |
| Augmented Training | 94.0% | 28.4% | 26.0% |

## 8.6 Results Analysis

The experiment reveals that the simple adversarial augmentation configuration did not improve robustness.

Compared with the baseline model:

- Clean accuracy decreased from 96.2% to 94.0%.
- FGSM accuracy decreased from 51.6% to 28.4%.
- PGD accuracy decreased from 48.9% to 26.0%.

This demonstrates that simply adding adversarial examples to the training dataset does not necessarily improve adversarial robustness.

The augmented model performed worse under both evaluated attacks.

Possible factors to investigate include:

- The adversarial training perturbation strength.
- The proportion of clean and adversarial training samples.
- The number and diversity of adversarial examples.
- The relationship between training and evaluation perturbation strengths.
- The optimization and model configuration.

These are investigation directions, not proven causes of the observed performance.

## 8.7 Robustness Decision Memo

### Recommended Decision: Improve Before Deployment

Based on the recorded results, the augmented model should not be accepted as a successful defensive improvement.

The model demonstrates lower clean accuracy and lower adversarial accuracy than the baseline under the tested evaluation conditions.

The recommended decision is to improve the defensive training strategy and repeat the robustness assessment before considering deployment.

Recommended next steps:

1. Evaluate multiple adversarial training epsilon values.
2. Compare different clean-to-adversarial training sample ratios.
3. Test multiple augmentation configurations.
4. Evaluate each configuration against FGSM and PGD.
5. Monitor clean accuracy and adversarial accuracy together.
6. Retain the baseline as a reference for every defensive experiment.

**Decision rationale:** A defensive method should demonstrate measurable improvement against relevant attacks without unacceptable degradation in clean performance. The current augmentation configuration does not meet that objective.

---

# 9. Consolidated Experimental Results

The following table summarizes the main results across all four laboratories.

| Experiment | Configuration | Recorded Result |
|---|---|---:|
| Baseline Classification | Clean test | 96.22% |
| FGSM Evasion | Epsilon 0.03 | 90.7% |
| FGSM Evasion | Epsilon 0.07 | 79.1% |
| FGSM Evasion | Epsilon 0.12 | 51.6% |
| FGSM Evasion | Epsilon 0.18 | 20.4% |
| Label Poisoning | 20% corruption | 93.3% |
| Label Poisoning | 50% corruption | 86.0% |
| Label Poisoning | 80% corruption | 54.2% |
| Label Poisoning | 100% corruption | 0.7% |
| Baseline Model | PGD evaluation | 48.9% |
| Augmented Model | Clean evaluation | 94.0% |
| Augmented Model | FGSM evaluation | 28.4% |
| Augmented Model | PGD evaluation | 26.0% |

**Interpretation:** The experiments identify vulnerabilities in both input handling and training-data integrity. The tested augmentation strategy also illustrates that defensive techniques must be empirically validated rather than assumed to be effective.

---

# 10. Security Findings

The four laboratories demonstrate several important machine learning security principles.

### 10.1 Clean Accuracy Is Not Robustness

A model can achieve high accuracy on normal data while remaining vulnerable to adversarial input perturbations.

### 10.2 Input Manipulation Can Degrade Predictions

FGSM experiments show a significant reduction in model accuracy as perturbation strength increases.

### 10.3 Training Data Integrity Is Critical

Random label corruption can severely affect model performance, demonstrating the importance of dataset provenance, integrity validation, and controlled modification privileges.

### 10.4 Defensive Retraining Requires Adaptive Evaluation

Adversarial augmentation does not automatically improve robustness. A defensive model must be tested against attacks generated using that model's own parameters.

### 10.5 Security Requires End-to-End Protection

Machine learning security should consider the complete lifecycle:

- Dataset acquisition and provenance.
- Training data integrity.
- Model development and training.
- Prediction input validation.
- Adversarial robustness testing.
- Defensive evaluation and monitoring.
- Model versioning and auditability.

---

# 11. Limitations

The experiments have several limitations:

1. The dataset consists of small 8 × 8 grayscale handwritten digit images.
2. The experiments use Logistic Regression and do not represent every modern machine learning architecture.
3. FGSM and PGD evaluations assume white-box access to the locally trained model.
4. The poisoning experiment uses random label corruption rather than targeted poisoning or backdoor attacks.
5. The PGD implementation uses a limited attack budget of seven iterations.
6. The adversarial augmentation strategy is a simple introductory implementation.
7. The experiments do not establish robustness against every possible adversarial attack.
8. The results cannot be generalized automatically to production AI systems.

These limitations define the scope of the controlled laboratory environment.

---

# 12. Repository Structure

```text
AICS202-Adversarial-Machine-Learning/
│
├README.md
│
├
│── AICS202_Lab01_Baseline_Threat_Model.ipynb
│── AICS202_Lab02_FGSM_Evasion.ipynb
│── AICS202_Lab03_Poisoning_Data_Integrity.ipynb
│── AICS202_Lab04_Robustness_Defences.ipynb
│
├
│── baseline_confusion_matrix.png
│── fgsm_robustness_curve.png
│── adversarial_examples.png
│── poisoning_accuracy_curve.png
│── defensive_comparison.png
│
│── AICS202_Final_Report.pdf
│
├
└── requirements.txt
```

*Note: The structure above is the recommended repository layout. Export the figures and create the documentation and PDF reports before claiming that all listed files are present.*

---

# 13. Reproducibility

To reproduce the experiments:

1. Clone or download the repository.
2. Install the required Python dependencies.
3. Open the Laboratory 01 notebook.
4. Execute all cells to establish the baseline model.
5. Open Laboratory 02 and execute the FGSM experiments.
6. Open Laboratory 03 and execute the label-corruption simulation.
7. Open Laboratory 04 and execute adversarial augmentation and robustness evaluation.
8. Verify all printed results and generated visualizations.
9. Export the relevant charts and reports.

### Random Seeds

| Experiment | Random Seed |
|---|---:|
| Dataset Split | 42 |
| Model Configuration | 42 |
| Label Corruption Generator | 2026 |
| PGD Initialization | 42 |

Using fixed seeds supports reproducibility, although results may still vary across software versions and execution environments.

---

# 14. Conclusion

This project provides a practical investigation of adversarial machine learning security across four laboratory modules.

Laboratory 01 establishes a Logistic Regression baseline with 96.22% clean-test accuracy.

Laboratory 02 demonstrates that FGSM perturbations can substantially reduce classification performance, reaching 20.4% accuracy at epsilon 0.18.

Laboratory 03 demonstrates the impact of training-label corruption, with accuracy decreasing to 0.7% when all training labels are corrupted.

Laboratory 04 evaluates a simple adversarial augmentation strategy. The recorded results show that the augmented model performs worse than the baseline under the tested clean, FGSM, and PGD conditions.

The overall findings emphasize three critical principles:

1. High clean-test accuracy does not guarantee adversarial robustness.
2. Training-data integrity is fundamental to machine learning security.
3. Defensive techniques must be validated through rigorous, adaptive evaluation.

The experiments provide a foundation for further research into adversarial machine learning, secure data pipelines, robust model training, and AI security assurance.

---

# 15. References

1. Scikit-learn — [Optical Recognition of Handwritten Digits Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html)
2. Scikit-learn — [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
3. Scikit-learn — [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
4. Goodfellow, I. J., Shlens, J., & Szegedy, C. — [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
5. ICDFA — AICS-202 Adversarial Machine Learning Training Manual.

---

**Project:** ICDFA AICS-202 – Adversarial Machine Learning  
**Laboratories:** 01–04  
**Dataset:** Scikit-learn Digits  
**Baseline Model:** Logistic Regression  
**Attack Techniques:** FGSM and PGD  
**Poisoning Simulation:** Random Training-Label Corruption  
**Defensive Method:** FGSM Adversarial Augmentation  
**Execution Scope:** Controlled Local Laboratory Environment  
**Documentation:** Consolidated Laboratory README
