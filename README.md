# YouTube-Sentiment-Insights

## Prerequisites
- Python 3.12+ (tested on Windows with Python 3.12.7)

### Microsoft C++ Build Tools (required for some packages)
Some dependencies (e.g., `ruamel.yaml.clibz` pulled by DVC) may require compilation on Windows.
- install: Visual Studio Build Tools 2022

## Setup (Windows / Command Prompt)
:: 1) Create venv
py -3.12 -m venv .venv

:: 2) Activate
.\.venv\Scripts\activate

:: 3) Upgrade pip tools
python -m pip install --upgrade pip setuptools wheel

:: 4) Install deps
pip install -r requirements.txt

:: 5) (If you have -e . in requirements, this installs your package too)


## Initialize DVC
dvc init

dvc repro

dvc dag


## AWS
 
aws configure

## Project Structure Tree
project_root/
├── params.yaml              # Your configuration file
├── dvc.yaml                 # If you are using DVC pipelines
├── data/
│   ├── raw/                 # Original raw_dataset.csv
│   └── processed/           # train.csv, test.csv
├── src/
│   ├── __init__.py
│   ├── data/                # <--- YOUR CURRENT FOCUS
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Loads CSV, splits Train/Test, saves to disk
│   │   └── data_preprocessing.py # Text cleaning (regex, lemmatization, stop words)
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # TF-IDF Vectorization logic
│   └── models/
│       ├── __init__.py
│       └── train_model.py       # Stacking Ensemble l