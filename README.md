# Social Sentiment MLOps Pipeline (Core Engine)

An end-to-end MLOps project implementing a **Stacking Ensemble Classifier** (LightGBM + Logistic Regression + KNN) to predict sentiment in social media comments. While currently trained on Reddit data, this engine is designed as the core processing unit for a broader YouTube/Social media insight system.

This project demonstrates a production-ready workflow using **DVC (Data Version Control)** for reproducibility, **NLTK** for advanced text processing, and a modular architecture.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Project Architecture](#project-architecture)
- [Running the Pipeline](#running-the-pipeline)
- [Inference & Prediction](#inference--prediction)
- [Pipeline Stages](#pipeline-stages)
- [Results](#results)


## Prerequisites

* **Python 3.12+** (Tested on Windows 11 with Python 3.12.7)
* **Microsoft C++ Build Tools**: Required for compiling specific Python dependencies (e.g., `ruamel.yaml` used by DVC).
    * *Install "Desktop development with C++" via Visual Studio Build Tools 2022.*


## Installation & Setup

These instructions are for **Windows (Command Prompt / PowerShell)**.

### 1. Environment Setup
Create and activate a virtual environment to keep dependencies isolated.

:: Create virtual environment

    - py -3.12 -m venv .venv

:: Activate the environment

    - .\.venv\Scripts\activate

### 2. Install Dependencies
Update your package tools and install the required libraries.

:: Upgrade core 

    - python -m pip install --upgrade pip setuptools wheel

:: Install project requirements

    - pip install -r requirements.txt


## Project Architecture

    ├── data/                           # Data Store
    │   ├── raw/                        # Original immutable datasets
    │   ├── ingested/                   # Split datasets (train.csv, test.csv)
    │   └── processed/                  # Cleaned text & TF-IDF matrices (.pkl)
    │
    ├── models/                         # Model Registry (Artifacts)
    │   └── stacking_model.pkl          # The trained Stacking Ensemble model
    │
    ├── src/                            # Source Code
    |   ├── __init__.py                 <-- REQUIRED
    │   ├── data/
    |   |   ├── __init__.py             <-- REQUIRED (Allows: from src.data import ...)
    │   │   ├── data_ingestion.py       # Loads raw data, splits Train/Test
    │   │   └── data_preprocessing.py   # Cleaning, Lemmatization, Stopwords
    │   ├── features/
    │   │   ├── __init__.py             <-- REQUIRED (Allows: from src.features import ...)
    │   │   └── build_features.py       # TF-IDF Vectorization (Trigrams, 10k features)
    │   └── models/
    │   |    ├── __init__.py            <-- REQUIRED (Allows: from src.models import ...)
    │   |    ├── train_model.py         # Stacking Ensemble training logic
    │   |    └── evaluate_model.py      # Metrics calculation (Accuracy, F1)
    |   | 
    |   └── prediction/
    |        ├── __init__.py            <-- REQUIRED (Allows: from src.prediction import ...)
    |        └── predict_model.py
    │
    ├── dvc.yaml                        # DVC Pipeline Definitions
    ├── params.yaml                     # Central Configuration (Hyperparameters)
    └── metrics.json                    # Model Performance Metrics


## Running the Pipeline
This project uses DVC (Data Version Control) to manage the ML pipeline.

1. Initialize (First Run Only)
If you are setting this up for the first time:

    * git init  (git bash)
    * dvc init

2. Reproduce Pipeline
Run the entire end-to-end workflow (Ingestion → Preprocessing → Training → Evaluation). DVC will only run stages that have changed.

    * dvc repro

3. DVC Diagram (Pipelines)
* dvc dag

4. View Metrics
Check the performance of the trained model.

    * dvc metrics show


## Inference & Prediction
Once the pipeline has run and the model is trained, you can use the predictor script to classify new text.

**Interactive Mode**

Launch a chat-like interface to test sentences rapidly.

    python src/prediction/predict_model.py
*Type "exit" to quit.*

**Single Command Mode**

Pass a string directly via the command line.

    python src/prediction/predict_model.py --text "This end-to-end MLOps project was incredibly helpful!"

## Pipeline Stages
1. Ingestion:

    - Loads raw Reddit/YouTube data.

    - Splits into Training (80%) and Testing (20%) sets using stratified sampling.

2. Preprocessing:

    - Cleaning: Regex removal of HTML tags, URLs, and punctuation.

    - Normalization: NLTK-based Lemmatization and custom Stopword removal.

3. Featurization:

    - TF-IDF Vectorizer: Generates 10,000 features.

    - N-Grams: Uses Unigrams, Bigrams, and Trigrams (1,3) to capture context.

4. Training:

    - Algorithm: Stacking Ensemble Classifier.

    - Base Learners: LightGBM (Gradient Boosting) + Logistic Regression (Balanced).

    - Meta Learner: K-Nearest Neighbors (KNN).

5. Evaluation:

    - Calculates Accuracy, Precision, Recall, and F1-Score (Weighted).


## Results
Current Champion Model Performance on Test Data:

    Metric 	      Score
    Accuracy	  86.82%
    F1 Score	  86.71%
    Precision	  86.76%
    Recall	      86.82%

*Note: Metrics are tracked automatically in metrics.json via DVC.*