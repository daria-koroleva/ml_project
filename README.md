# **📧 Phishing Email Detector**

**Course:** COMP 6321: Machine Learning
**Team Members:**

* **Hadi El Nawfal**
* **Daria Koroleva**
* **Khalil Nijaoui**

## **📌 Project Overview**

This project builds and evaluates phishing-email detection models using both **traditional machine learning** and a **compact transformer model (TinyBERT)**.
All models are trained and evaluated on the same **Hugging Face dataset**, using an identical **30% / 35% / 35%** stratified split.

The complete methodology, results, and analysis are documented in the final report.

## **📂 Dataset**

We use the **compiled-phishing-dataset**:

👉 [https://huggingface.co/datasets/renemel/compiled-phishing-dataset](https://huggingface.co/datasets/renemel/compiled-phishing-dataset)

**Dataset summary:**

* **119,148 emails**
* Binary labels: *phishing* or *legitimate*
* Raw email text
* Additional metadata such as:

  * sender domain
  * word count
  * sentence count
  * average words per sentence

## **🧹 Data Preprocessing**

Run:

```
data_preprocessing_new.ipynb
```

This script:

* Loads the dataset
* Generates structural features
* Creates a deterministic stratified split:

| Split          | Size                |
| -------------- | ------------------- |
| **Train**      | 30% (35,744 emails) |
| **Validation** | 35% (41,702 emails) |
| **Test**       | 35% (41,702 emails) |

Generated files placed in `/data`:

* `train.csv`
* `val.csv`
* `test.csv`


## **🧠 Models Implemented**

### **1. Traditional Machine Learning Models**

All traditional models share the same input features:

✔ **TF-IDF (1–2 grams, up to 100k features)**
✔ **Structural features:**

* word count
* sentence count
* words per sentence

✔ **Categorical domain feature (one-hot)**
✔ **GridSearchCV (3-fold, F1-score)**

**Models:**

| Model                       | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| **Logistic Regression**     | Linear classifier with great results for low resources  |
| **Linear SVM**              | Margin-based linear model, best overall performance     |
| **Random Forest**           | Non-linear tree ensemble                                |
| **Multinomial Naive Bayes** | Probabilistic text classifier                           |

Run:

```
traditional_models.ipynb
```

This notebook performs:

* full training
* hyperparameter tuning
* validation + test evaluation
* confusion matrices
* saved models

---

### **2. Transformer Model — TinyBERT**

Fine-tuned using Hugging Face Transformers.

**Configuration:**

* **Model:** TinyBERT
* **Tokenizer:** WordPiece
* **Max length:** 128
* **Epochs:** 10
* **Batch size:** 64
* **Framework:** PyTorch
* **Hardware:** GPU (RTX 3090)

Run:

```
tinyBERT_explainable.ipynb
```

Outputs:

* fine-tuned model checkpoint
* metrics
* confusion matrix
* precision-recall curves

---

## **📊 Results Summary**

From the full report ****, all models are evaluated using the **same test set**.

### **Test-Set Performance**

| Model                   | Accuracy | Precision | Recall    | F1        | PR–AUC |
| ----------------------- | -------- | --------- | --------- | --------- | ------ |
| **Logistic Regression** | 0.987    | 0.992     | 0.986     | **0.989** | 0.999  |
| **Linear SVM**          | 0.988    | 0.992     | 0.986     | **0.989** | 0.999  |
| **Random Forest**       | 0.984    | 0.986     | 0.986     | 0.986     | 0.998  |
| **Naive Bayes**         | 0.979    | 0.980     | 0.983     | 0.982     | 0.998  |
| **TinyBERT**            | 0.940    | 0.943     | **0.940** | 0.939     | 0.989  |

### **Interpretation**

* **Linear models (LR, SVM)** achieve the best overall performance.
* **TinyBERT** achieves **excellent recall for phishing emails** (fewer missed attacks) but has more false positives.
* **Random Forest** trades precision for recall, consistent with non-linear decision boundaries.
* **Naive Bayes** performs strongly despite simplicity.

---

## **📦 Reproducibility**

All experiments use:

* `random_state = 42`
* identical preprocessing
* identical train/val/test split
* full hyperparameter search grids
* saved models and evaluation reports

See full methodology in the attached report ****.

---

## **🚀 How to Run Everything**

1. **Clone the repository**

   ```bash
   git clone <repo_url>
   cd ml_project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ````
3. Run preprocessing

   ```bash
   jupyter notebook data_preprocessing_new.ipynb
   ```

4. Run traditional models

   ```bash
   jupyter notebook traditional_models.ipynb
   ```

5. Run TinyBERT Transformer

   ```bash
   jupyter notebook tinyBERT_explainable.ipynb
   ```

All results will be stored under:

```
/models/
/results/
```

---

## **📘 Reference**

Full report: ****
Dataset: [https://huggingface.co/datasets/renemel/compiled-phishing-dataset](https://huggingface.co/datasets/renemel/compiled-phishing-dataset)
