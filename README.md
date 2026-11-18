# AICS-202-Integrated-Adversarial-Machine-Learning-Practical-Investigation
AICS-202 · Practical Laboratories Due · Integrated Adversarial Machine Learning Practical Investigation
- Dataset: sklearn.datasets.load_digits
- Dataset shape: (1797, 64)
- Training samples: 1347
- Testing samples: 450
- Clean test accuracy: 0.9622 atau 96.22%
- Model: Logistic Regression (sesuai konfigurasi baseline pada praktikum).
1. Completed Threat Model
Field	Student response
Protected model and asset	The locally trained Logistic Regression classifier and its prediction integrity, model parameters, training data, and the reliability of digit classification results.
Legitimate model function	To classify handwritten digit images (0–9) using 64 numerical pixel features derived from 8×8 grayscale images.
Potential attacker-controlled input	Digit image inputs submitted for prediction. An attacker may manipulate pixel values while keeping the input within the valid range of 0–1 to influence the model's prediction.
White-box or black-box assumption	White-box assumption for this practical exercise. The attacker is assumed to know the model architecture, parameters, preprocessing method, and input representation, allowing gradient-based adversarial attacks such as FGSM.
Desired adversarial outcome	Cause the classifier to incorrectly classify a digit by applying small, controlled perturbations to the input image while attempting to preserve its original visual meaning.
Likely monitoring / prevention controls	Input validation and range checking, adversarial input testing, monitoring prediction confidence and abnormal input patterns, adversarial training, robustness evaluation using FGSM and PGD, model version control, and maintaining reproducible experiment logs.

2. Reflection
Question: Why does good clean-test accuracy not establish security against manipulated inputs?

A clean-test accuracy of 96.22% indicates that the Logistic Regression model performs well on normal, unmodified test samples. However, this metric does not demonstrate that the model is robust against adversarial manipulation.

Adversarial attacks exploit the model's sensitivity to small changes in input features. An attacker can deliberately modify pixel values within the permitted input range to create an adversarial example that causes the classifier to produce an incorrect prediction, even when the original image is correctly classified.

For example, an image representing digit 3 may be modified with carefully calculated pixel perturbations so that the model predicts another digit, such as 8. Although the changes may be small and difficult to notice visually, they can significantly affect the model's prediction.

Clean-test accuracy evaluates performance under normal data conditions, whereas adversarial robustness evaluates how well the model maintains its prediction performance under deliberately manipulated inputs.

Therefore, a high clean-test accuracy should not be interpreted as evidence of security. Additional security evaluation is necessary, including FGSM and PGD attacks, robustness measurements at different perturbation levels, and defensive techniques such as adversarial training.

Conclusion: Model accuracy and model security are two different properties. A model can achieve excellent accuracy on clean data while remaining vulnerable to adversarial attacks.
