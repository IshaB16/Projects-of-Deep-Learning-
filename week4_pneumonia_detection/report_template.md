# Week 4 Deep Learning Capstone Report

## 1. Objective
Classify chest X-ray images into NORMAL and PNEUMONIA for an educational deep-learning experiment.

## 2. Dataset and ethics
- Dataset source and license:
- Image counts by split and class:
- Class balance, provenance, privacy, and possible bias:

## 3. Preprocessing and augmentation
Describe resizing, preprocessing, augmentation, and split strategy. Keep the test set untouched.

## 4. Model architecture
Describe MobileNetV2 transfer learning, global average pooling, dropout, and sigmoid output.

## 5. Experiments
| Experiment | Learning rate | Batch size | Dropout | Epochs | Fine-tuning |
|---|---:|---:|---:|---:|---|
| A | 0.0001 | 16 | 0.25 | 5 | No |
| B | | | | | |

## 6. Results
Use actual values from `reports/metrics.json` and `reports/classification_report.txt`. Do not invent metrics.

## 7. Confusion matrix and ROC curve
Insert generated plots and explain false positives and false negatives.

## 8. Deployment
Describe the Streamlit UI, Flask API, and Docker steps.

## 9. Limitations and responsible use
Discuss dataset bias, representativeness, uncertainty, external validation, and why this prototype is not for medical decisions.

## 10. Conclusion and future work
Summarize actual findings and improvements.

## 11. References
List the dataset source, associated research paper, and software documentation.
