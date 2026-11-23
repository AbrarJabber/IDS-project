# Model figure notes

- **01_model_comparison.png**: Attack precision, recall and F1 are positive-class metrics. Both RF thresholds use the same fitted forest.

- **02_confusion_matrices.png**: Rows represent actual classes. Colors share a 0–100% scale. Percentages use each actual class total.

- **03_overlap_comparison.png**: The full test set contains both groups. Differences are descriptive, not a causal overlap effect. Lower false-positive rate is better; higher accuracy and F1 are better.

- **04_attack_family_results.png**: The model is binary. Family-level analysis uses known test annotations. Small families have limited sample support.