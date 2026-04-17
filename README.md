# Automating Ticket Routing with Machine Learning

An end-to-end machine learning system designed to automatically classify incoming customer support tickets into the correct department queues. This system reduces manual triage time and ensures tickets reach the right experts instantly.

## Overview

This project uses Natural Language Processing (NLP) to analyze ticket subjects and bodies to predict one of 10 department categories. It includes a full pipeline from data preprocessing and exploratory data analysis (EDA) to model training and persistence.

### Key Features
- **Intelligent Preprocessing**: Custom NLP pipeline for cleaning raw ticket text (regex cleaning, stopword removal, lemmatization).
- **TF-IDF Vectorization**: Converts text data into numerical features using unigrams and bigrams.
- **Persistent Models**: Fully trained models saved via `joblib` for immediate inference.
- **Production-Ready**: Designed for easy integration into existing helpdesk workflows.

## 📁 Project Structure

```text
Ticketing_project/
├── data/                    # Raw and processed datasets
├── notebooks/               # Development and demonstration notebooks
│   ├── model.ipynb          # Training pipeline & EDA
│   ├── try.ipynb            # Inference playground
│   ├── ticket_classifier.joblib # Saved scikit-learn pipeline
│   ├── le.joblib            # Saved LabelEncoder for category mapping
│   └── tickets.csv          # Training dataset
├── .gitignore               # Environment and data exclusions
└── README.md                # Project documentation
```

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Ticketing_project
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv env
    # On Windows:
    .\env\Scripts\activate
    # On macOS/Linux:
    source env/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk joblib
    ```

4.  **Download NLTK data** (first-time use):
    ```python
    import nltk
    nltk.download(['stopwords', 'wordnet', 'punkt'])
    ```

## Model Details

### Preprocessing
The system implements a robust `preprocess_text` function that:
- Fixes escaped newlines.
- Removes URLs and Email addresses.
- Strips punctuation and digits.
- Removes common english stopwords.
- Performs **Lemmatization** via NLTK's `WordNetLemmatizer`.

### Pipeline
The model is built as a Scikit-Learn `Pipeline` which includes:
- `TfidfVectorizer`: 50,000 max features, (1, 2) n-gram range.
- `Classifier`: Trained to predict across 10 distinct departments including *Technical Support*, *Billing*, *Returns*, and *Sales*.

## Usage

To use the model in your own script:

```python
import joblib

# Load components
model = joblib.load('notebooks/ticket_classifier.joblib')
le = joblib.load('notebooks/le.joblib')

# Raw ticket data
raw_text = "I am having trouble accessing the invoice for my last purchase"

# Predict (Note: Model expects an iterable/list)
prediction_idx = model.predict([raw_text])
category = le.inverse_transform(prediction_idx)

print(f"Assigned Queue: {category[0]}")
```

## 📊 Dataset
The model was trained on `tickets.csv` containing ~13,700 tickets across 10 categories:
- Technical Support
- Returns and Exchanges
- Billing and Payments
- Sales and Pre-Sales
- IT Support
- General Inquiry
- Customer Service
- Product Support
- Service Outages and Maintenance
- Human Resources

---

