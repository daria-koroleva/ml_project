# Phishing Email Detector

**Course:** COMP 6321 

**Team:**  
- Hadi El Nawfal
- Daria Koroleva
- Khalil Nijaoui

## Data Setup Instructions

1. Download the  dataset from https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

2. Create folder inside your project directory **/data/raw/** and place 6 csv files there:
**Note: make sure to exclude phishing_email.csv**
      - CEAS.csv
      - Enron.csv
      - Ling.csv
      - Nazario.csv
      - Nigerian_Fraud.csv
      - SpamAssasin.csv
3. Run the script
4. Clean data will be saved under **/data/preprocessing/**
5. By the end of the script, you supposed to have 5 files:
    - merged_raw.csv
    - phishing_full_clean.csv
    - train.csv
    - val.csv
    - test.csv
    
## Models
- **Model 1 (Baseline):**  TF-IDF (bag-of words and tokenization) + Logistic Regression.
run logistic_regression.ipynb to obtain all scores.
- **Model 2 (Transformer):** DistilBERT.

