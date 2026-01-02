import os
import re
import yaml
import logging
import argparse
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. NLTK Resource Handling (Run Once)
def ensure_nltk_resources():
    """
    Checks if necessary NLTK data exists; downloads if missing.
    """
    resources = ['corpora/stopwords', 'corpora/wordnet']
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            resource_name = resource.split('/')[1]
            logging.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name)

ensure_nltk_resources()

# 3. Global NLP Objects
# Initialize these once to avoid reloading them for every single row (Huge performance boost)
lemmatizer = WordNetLemmatizer()
# Custom stopword list: standard English minus specific words valuable for sentiment
STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

def load_params(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_text(text, remove_html=True, remove_urls=True, lower=True):
    """
    Advanced text cleaning pipeline.
    Combines Regex cleaning (HTML/URL/Punctuation) with NLTK (Stopwords/Lemmatization).
    """
    if not isinstance(text, str):
        return ""
    
    # ---Part A: Regex & Basic Cleaning ---
    # 1. Lowercase
    if lower:
        text = text.lower()
    
    # 2. Remove URLs
    if remove_urls:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove HTML tags
    if remove_html:
        text = re.sub(r'<.*?>', '', text)

    # 4. Remove Punctuation
    # We remove punctuation before splitting to ensure "word," becomes "word"
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Remove Newlines/Tabs
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # ---Part B: NLTK Processing (Stopwords & Lemmatization)---
    # 6. Tokenize (split by whitespace)
    tokens = text.split()
    
    # 7. Remove Stopwords & Lemmatize
    # This list comprehension does both steps efficiently
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in STOP_WORDS
    ]

    # 8. Join back to string
    text = ' '.join(processed_tokens)

    return text

def preprocess_data(config_path):
    params = load_params(config_path)
    
    # Path Setup
    base_dir = params['data_ingestion']['processed_dir']
    prep_config = params['preprocessing']
    
    train_input = os.path.join(base_dir, prep_config['input_train_file'])
    test_input = os.path.join(base_dir, prep_config['input_test_file'])
    
    train_output = os.path.join(base_dir, prep_config['output_train_file'])
    test_output = os.path.join(base_dir, prep_config['output_test_file'])

    logging.info("Starting text preprocessing with NLTK...")

    # Process files
    for input_path, output_path, label in [(train_input, train_output, 'Train'), (test_input, test_output, 'Test')]:
        try:
            logging.info(f"Reading {label} data from {input_path}")
            df = pd.read_csv(input_path)

            # Ensure text column is string
            # Assuming column name is 'clean_comment' from ingestion
            if 'clean_comment' not in df.columns:
                raise ValueError(f"Column 'clean_comment' not found in {input_path}")

            # Apply cleaning
            logging.info(f"Cleaning {label} text...")
            df['clean_comment'] = df['clean_comment'].apply(
                lambda x: clean_text(
                    x, 
                    remove_html=prep_config['remove_html'], 
                    remove_urls=prep_config['remove_urls'],
                    lower=prep_config['lower_case']
                )
            )
            
            # Remove any rows that became empty after cleaning
            initial_len = len(df)
            df = df[df['clean_comment'] != ""]
            if len(df) < initial_len:
                logging.info(f"Removed {initial_len - len(df)} rows that were empty after cleaning.")

            # Save
            df.to_csv(output_path, index=False)
            logging.info(f"Saved cleaned {label} data to {output_path}")

        except FileNotFoundError:
            logging.error(f"File not found: {input_path}. Did you run data_ingestion.py?")
            raise
        except Exception as e:
            logging.error(f"Error processing {label} data: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    preprocess_data(config_path=args.config)