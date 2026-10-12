# Explainable Deep Learning for Pneumonia Detection Using Chest X-Ray Images

A research project comparing explainability and classification performance for pediatric pneumonia detection using chest X-ray (CXR) images and a shared ResNet-50 backbone.

## Overview

Deep learning models can classify chest X-rays accurately while relying on irrelevant visual cues, such as borders, text markers, or other acquisition-related artifacts, instead of meaningful lung findings. This project investigates that problem by evaluating four explainable artificial intelligence (XAI) approaches in a controlled experimental setup. The aim is to consider both predictive performance and the spatial relevance of the explanations produced by the models.

The study compares a standard ResNet-50 baseline with four variants:

1. **Composite Layer-wise Relevance Propagation (LRP)**
2. **Gradient-weighted Class Activation Mapping (Grad-CAM)**
3. **Spatial Attention Mechanism (SAM)**
4. **Fast Gradient Sign Method (FGSM) adversarial training**

All variants use the same underlying ResNet-50 family of architecture so that differences can be examined in a common experimental setting. The project also evaluates attribution quality with the Mean Relevance Score (MRS), checks relevance conservation for LRP, and reports a Trade-off Score for SAM and adversarial training.

> **Research use only:** This repository describes an academic experiment. The models have not been established as clinically validated diagnostic tools and should not be used to make patient-care decisions.

## Research objectives

- Compare the classification performance of a ResNet-50 baseline and four XAI-related variants on pediatric CXR images.
- Inspect whether model explanations emphasize lung regions rather than irrelevant image borders or artifacts.
- Quantitatively evaluate attribution quality using MRS and, for the LRP implementation, conservation-error measurements.
- Compare classification outcomes and explanation quality with paired statistical tests.
- Examine the trade-off between prediction performance and explanation quality.

## Dataset

The experiment uses the publicly available [Chest X-Ray Images (Pneumonia) dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

| Item | Description |
|---|---|
| Image type | Pediatric chest X-ray images |
| Total images | 5,856 |
| Pneumonia class | 4,273 images |
| Normal class | 1,583 images |
| Task | Binary classification: `PNEUMONIA` vs. `NORMAL` |
| Model input | 224 × 224 pixels |

The thesis describes the source images as coming from a pediatric hospital and notes that the dataset documentation reports physician review of the radiographs. The dataset is imbalanced, with more pneumonia images than normal images.

### Data partitioning

The supplied dataset includes `train`, `val`, and `test` folders. In the thesis protocol, the original training and validation folders were combined and then split using stratified sampling into a new training partition and validation partition. The original test partition was held out for final evaluation.

| Partition | Number of images | Purpose |
|---|---:|---|
| Training | 4,709 | Model fitting |
| Validation | 523 | Monitoring, model selection, and threshold selection |
| Held-out test | 624 | Final evaluation |
| **Total** | **5,856** | |

The thesis reports that the held-out test partition contains 390 pneumonia images and 234 normal images. The held-out test set was not intended to be used for preprocessing decisions or threshold tuning.

## Preprocessing and class imbalance

The reported preprocessing pipeline includes:

- Resize images to **224 × 224** using bilinear interpolation.
- Normalize pixel intensity to the range **0 to 1** by dividing by 255.
- Apply a body crop intended to remove approximately 3% around the patient silhouette.
- Crop approximately 2% from each side to remove border artifacts, including scanner-imprinted text and orientation markers.
- Perform preprocessing before normalization in the image input pipeline.

The thesis reports inverse-frequency class weighting instead of oversampling or undersampling. The documented weights are approximately **1.866 for `NORMAL`** and **0.683 for `PNEUMONIA`**, calculated from the training partition. The thesis also mentions training-time augmentation, but the exact augmentation operations are not fully specified in the methods text summarized here. Check the original implementation before reproducing that part of the experiment.

A classification threshold was selected using the validation set to target a minimum sensitivity of 0.90, with the intention of reducing missed pneumonia cases. The held-out test data were reserved for final evaluation.

## Model architecture and training setup

ResNet-50 pretrained on ImageNet serves as the shared convolutional backbone. Its original ImageNet classification head was replaced by a binary pneumonia classifier with a sigmoid output. The Global Average Pooling (GAP) representation has 2,048 dimensions, and a dropout layer is applied before the classifier.

| Setting | Value reported in the thesis |
|---|---|
| Backbone | ImageNet-pretrained ResNet-50 |
| Input size | 224 × 224 pixels |
| Output | Binary classifier with sigmoid probability |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 32 |
| Maximum epoch budget | 50 |
| Dropout | 0.30 |
| L2 weight decay | 0.0001 |
| Loss | Binary cross-entropy |
| Learning-rate reduction | ReduceLROnPlateau, factor 0.5, patience 5, minimum learning rate 1e-7 |
| Early stopping | Patience 10 |
| Checkpoint criterion | Minimum validation loss |
| Hyperparameter search | None reported |
| SAM attention kernel | 7 × 7 |
| FGSM perturbation magnitude | ε = 0.01 |

The thesis states that the model layers were fine-tuned end-to-end. Experiments were conducted in Kaggle using Python and TensorFlow/Keras, with OpenCV, NumPy, Matplotlib, and scikit-learn also used in the workflow. Exact package versions are not specified in the thesis; a reproducible repository should record the versions used in its actual environment.

## Compared approaches

### 1. ResNet-50 baseline

The baseline is trained without one of the four additional XAI variants. It provides the reference classification performance for the comparisons.

### 2. Composite Layer-wise Relevance Propagation (LRP)

LRP propagates relevance backward through the network to estimate which input pixels contributed to a prediction. The thesis describes a composite implementation that accounts for ResNet residual connections and evaluates relevance conservation at multiple points in the network. MRS is used to quantify attribution quality within the lung region, while mean absolute error (MAE) is used for relevance-conservation checks.

### 3. Grad-CAM

Grad-CAM uses gradients and feature activations from the final convolutional layer, identified in the thesis as `conv5_block3_out`, to produce a class-specific localization heatmap. This is a post-hoc explanation method and does not add an attention block to the forward architecture.

### 4. Spatial Attention Mechanism (SAM)

SAM is integrated into the model. It calculates a spatial attention map from the ResNet-50 feature representation, using a 7 × 7 convolution and spatial softmax over the 7 × 7 feature grid. The resulting attention weights are intended to show which spatial locations contribute more strongly to the prediction.

### 5. FGSM adversarial training

The adversarial-training variant trains ResNet-50 on FGSM-perturbed images using a reported perturbation magnitude of ε = 0.01. Gradient saliency maps are used as the interpretability output. This variant changes the training procedure rather than simply applying an explanation method after training.

## Evaluation metrics

### Classification metrics

The thesis evaluates models using:

- **Accuracy:** proportion of all test images classified correctly.
- **Sensitivity (recall):** proportion of pneumonia cases correctly identified.
- **Specificity:** proportion of normal cases correctly identified.
- **Positive predictive value (PPV):** proportion of predicted pneumonia cases that are pneumonia cases.
- **Negative predictive value (NPV):** proportion of predicted normal cases that are normal cases.
- **F1-score:** harmonic mean of precision and recall.
- **AUC-ROC:** discrimination across classification thresholds.

### Explanation metrics and statistical tests

- **Mean Relevance Score (MRS):** the study's quantitative measure of the spatial relevance of an explanation with respect to the lung region.
- **LRP conservation error (MAE):** checks differences between relevance entering and leaving selected network boundaries.
- **Trade-off Score (TS):** reported for SAM and adversarial training, using the form `TS = Accuracy - λ × (1 - MRS)`. The value of λ should be taken from the actual implementation when reproducing this score.
- **Bootstrap confidence intervals:** used to summarize uncertainty in the reported evaluation metrics.
- **McNemar's test:** used for paired comparisons of classification outcomes on the same test images.
- **Wilcoxon signed-rank test:** used for paired comparisons of per-image relevance scores. The thesis reports Bonferroni correction for multiple pairwise comparisons.

## Main reported results

The following values are transcribed from the thesis's comparative results table. Metrics are reported as proportions; multiply by 100 to express them as percentages. MRS is not reported for the baseline.

| Model | Accuracy | Sensitivity | Specificity | PPV | NPV | F1-score | AUC-ROC | MRS (mean ± SD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 baseline | 0.9391 | 0.9667 | 0.8932 | 0.9378 | 0.9414 | 0.9520 | 0.9796 | Not reported |
| ResNet-50 + Composite LRP | 0.9391 | 0.9590 | 0.9060 | 0.9444 | 0.9298 | 0.9517 | 0.9824 | 0.9078 ± 0.0679 |
| ResNet-50 + Grad-CAM | **0.9503** | 0.9487 | **0.9530** | **0.9711** | 0.9177 | **0.9598** | **0.9872** | 0.7963 ± 0.0868 |
| ResNet-50 + SAM | 0.9375 | 0.9487 | 0.9188 | 0.9512 | 0.9149 | 0.9499 | 0.9794 | 0.6681 ± 0.1104 |
| ResNet-50 + FGSM adversarial training | 0.9279 | 0.9462 | 0.8974 | 0.9389 | 0.9091 | 0.9425 | 0.9745 | 0.8238 ± 0.0732 |

### Interpretation of the reported results

- **Grad-CAM** achieved the highest reported accuracy (**95.03%**) and AUC-ROC (**98.72%**) in the comparison table.
- **Composite LRP** achieved the highest reported MRS (**0.9078 ± 0.0679**) while matching the baseline accuracy at **93.91%**.
- **FGSM adversarial training** achieved an accuracy of **92.79%** and an MRS of **0.8238 ± 0.0732**. The thesis reports the smallest difference between pneumonia-class and normal-class MRS for this variant (0.0145).
- **SAM** achieved an accuracy of **93.75%**. Its overall MRS was the lowest of the four XAI variants (0.6681), with different class-specific scores reported for normal and pneumonia cases.
- The thesis reports that all six pairwise Wilcoxon comparisons of MRS were statistically significant after correction (reported p-values below 0.001). For classification outcomes, it reports a significant difference for Grad-CAM versus LRP (McNemar's χ² = 7.875, p = 0.005), with the remaining five pairwise comparisons not significant at the corrected threshold.

These are the results reported in the thesis and should not be interpreted as external validation. A common dataset and a single held-out test partition make the comparison internally controlled, but do not establish generalization to other hospitals or populations.

## Outputs described in the study

The thesis describes saving outputs from experimental runs in Kaggle, including:

- Training-history plots
- ROC curves and confusion matrices
- Explanation heatmaps and overlays for normal and pneumonia classes
- Per-run metric reports in JSON format
- Per-sample results in CSV format
- LRP conservation-error measurements and comparative metric tables

The actual notebook names, saved artifact paths, and package lock file should be documented in this repository once the executable experiment files are added.

## Reproducibility

The thesis documents the main model settings and evaluation design, but this README alone is not a runnable implementation. To reproduce the experiment, the repository should include the original training and evaluation notebook or scripts, a dependency file with exact tested versions, and instructions for downloading and arranging the dataset.

Suggested repository organization:

```text
.
├── README.md
├── notebooks/              # Add the experiment notebook(s)
├── src/                    # Add preprocessing, training, XAI, and evaluation code
├── results/                # Add metrics, confidence intervals, and statistical tests
├── figures/                # Add ROC curves, confusion matrices, and explanation maps
├── requirements.txt        # Pin versions from the tested environment
└── CITATION.cff            # Optional citation metadata
```

This is a suggested structure, not a claim that these files are included. Do not commit the full dataset unless its license and redistribution terms permit it; link to the original Kaggle dataset instead.

## Limitations

- The experiment uses one public pediatric CXR dataset from a single source.
- The reported test results do not establish performance across hospitals, scanners, demographics, or external datasets.
- MRS and heatmaps quantify or visualize aspects of attribution but do not prove that an explanation is clinically correct or that the model reasons causally.
- The binary task covers `NORMAL` and `PNEUMONIA` only.
- Exact software package versions and the complete executable code are not included in the thesis text provided for this README.
- Some implementation details, including the exact training-time augmentation operations and the Trade-off Score's λ value, should be checked against the original code before independent reproduction.

The thesis identifies future directions including testing on external datasets such as NIH ChestX-ray14 or CheXpert, extending the task to multiple classes, and exploring combinations of LRP and attention-based methods.

## Tools and technologies

- Python
- TensorFlow and Keras
- ResNet-50
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Kaggle notebook environment

Exact library versions should be added after checking the experiment environment.

## Citation

If you use or discuss this work, cite the thesis as follows:

> Ahmad, Adyan. *An Explainable Deep Learning Approach for Pneumonia Detection Using Chest X-Ray Images*. Final Year Project thesis, BS Computer Science, Department of Computer Science, Bacha Khan University, Charsadda, Pakistan, session 2022-2026.

Please replace or supplement this citation if a formal publication or DOI becomes available.

## Acknowledgment and data source

The experiment uses the publicly available [Kaggle Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Refer to the dataset page for its attribution and usage terms.

## Disclaimer

This project is for research and educational purposes only. It is not a medical device and is not intended to diagnose, treat, or prevent any disease. Clinical use would require appropriate external validation, clinical review, and relevant regulatory evaluation.
