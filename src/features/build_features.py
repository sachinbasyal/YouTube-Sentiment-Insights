import pandas as pd
import yaml
import logging
import argparse
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_features(config_path):
    params = load_params(config_path)
    
    base_dir = params['data_ingestion']['processed_dir']
    
    # Inputs
    input_train_path = os.path.join(base_dir, params['preprocessing']['output_train_file'])
    input_test_path = os.path.join(base_dir, params['preprocessing']['output_test_file'])
    
    # Outputs
    output_train_feat = os.path.join(base_dir, params['build_features']['output_train_features'])
    output_test_feat = os.path.join(base_dir, params['build_features']['output_test_features'])
    vectorizer_path = os.path.join(base_dir, params['build_features']['output_vectorizer'])

    # Settings
    max_features = params['build_features']['max_features']
    ngram_range = tuple(params['build_features']['ngram_range'])
    target_col = params['build_features']['target_col']  # <--- Read from params

    try:
        logging.info("Loading cleaned data...")
        train_df = pd.read_csv(input_train_path)
        test_df = pd.read_csv(input_test_path)

        # DEBUG: Check if column exists
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found. Available columns: {list(train_df.columns)}")

        # Handle NaNs in text
        train_df['clean_comment'] = train_df['clean_comment'].fillna('')
        test_df['clean_comment'] = test_df['clean_comment'].fillna('')

        # Separate X (Text) and y (Label)
        logging.info(f"Using target column: {target_col}")
        X_train_text = train_df['clean_comment']
        y_train = train_df[target_col]  # <--- Use variable, not hardcoded string
        
        X_test_text = test_df['clean_comment']
        y_test = test_df[target_col]

        # Vectorization
        logging.info(f"Initializing TfidfVectorizer (max_features={max_features}, ngram={ngram_range})...")
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        logging.info("Fitting vectorizer...")
        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)

        # Save Vectorizer
        logging.info(f"Saving vectorizer to {vectorizer_path}...")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        # Save Data
        logging.info("Saving feature matrices...")
        data_train = {'X': X_train_tfidf, 'y': y_train}
        data_test = {'X': X_test_tfidf, 'y': y_test}

        with open(output_train_feat, 'wb') as f:
            pickle.dump(data_train, f)
            
        with open(output_test_feat, 'wb') as f:
            pickle.dump(data_test, f)

        logging.info("Feature engineering completed successfully.")

    except Exception as e:
        logging.error(f"Error during feature building: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    build_features(config_path=args.config)